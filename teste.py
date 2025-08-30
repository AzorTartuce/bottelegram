import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import ccxt.async_support as ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
import asyncio
import os
from dotenv import load_dotenv
import logging
import json
import pickle
import base64 # Para codificar os modelos para o banco de dados
from supabase import create_client, Client # Importado para usar o Supabase

# Configuração de logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
# Carregando credenciais do Supabase do arquivo .env
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Inicializa o cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Cliente Supabase inicializado.")

# --- Funções de Coleta e Features (sem alterações) ---

async def fetch_crypto_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        await exchange.load_markets()
        if symbol not in exchange.markets:
            logger.error(f"Símbolo {symbol} não encontrado na Kraken.")
            await exchange.close()
            return None
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        await exchange.close()
        return df
    except ccxt.BadSymbol as e:
        logger.error(f"Erro de Símbolo Inválido ao coletar dados para {symbol}: {e}")
        if 'exchange' in locals(): await exchange.close()
        return None
    except Exception as e:
        logger.error(f"Erro ao coletar dados para {symbol}: {e}")
        if 'exchange' in locals(): await exchange.close()
        return None

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    if loss.eq(0).all(): return pd.Series(100.0, index=series.index)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def compute_momentum(series, period=10):
    return series - series.shift(period)

def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=7).std()
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['bb_upper'], df['bb_lower'] = compute_bollinger_bands(df['close'])
    df['vwap'] = compute_vwap(df)
    df['momentum'] = compute_momentum(df['close'])
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_risk_levels(last_close, atr, risk_factor=1.5, reward_factor=2.0):
    stop_loss = last_close - (atr * risk_factor)
    take_profit = last_close + (atr * reward_factor)
    return stop_loss, take_profit

# --- Funções de Treino e Previsão (Adaptadas para o Supabase) ---
# SQL para criar a tabela no Supabase SQL Editor:
# CREATE TABLE trained_models (
#     symbol TEXT PRIMARY KEY,
#     model_b64 TEXT NOT NULL,
#     scaler_b64 TEXT NOT NULL,
#     metrics_json TEXT NOT NULL,
#     last_trained_utc TIMESTAMPTZ NOT NULL
# );

async def train_and_save_model(symbol: str):
    logger.info(f"Iniciando processo de treino para {symbol}...")
    df = await fetch_crypto_data(symbol=symbol, limit=1000)
    if df is None or df.empty:
        return False, "Erro ao obter dados para o treino."

    df_features = create_features(df)
    
    features = ['open', 'high', 'low', 'volume', 'returns', 'log_returns', 'volatility', 
                'ma7', 'ma21', 'ma50', 'ma200' ,'rsi', 'atr', 'vwap', 'momentum']
    X = df_features[features]
    y = df_features['target']

    if len(X) < 100:
        return False, "Dados insuficientes para um treino confiável."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, 
                                   random_state=42, class_weight='balanced', n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    model_b64 = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    scaler_b64 = base64.b64encode(pickle.dumps(scaler)).decode('utf-8')
    
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision_up": float(report.get("1", {}).get("precision", 0)),
        "precision_down": float(report.get("0", {}).get("precision", 0)),
    }
    
    data_to_upsert = {
        "symbol": symbol,
        "model_b64": model_b64,
        "scaler_b64": scaler_b64,
        "metrics_json": json.dumps(metrics_dict),
        "last_trained_utc": pd.Timestamp.now(tz='UTC').isoformat()
    }

    try:
        # CORREÇÃO: A função upsert espera uma LISTA de dicionários.
        response = supabase.table('trained_models').upsert([data_to_upsert]).execute()
        if len(response.data) == 0:
            raise Exception(f"Falha no upsert, nenhum dado retornado. Resposta: {response}")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo no Supabase para {symbol}: {e}")
        return False, "Falha ao salvar o modelo treinado no banco de dados."
        
    logger.info(f"Treino para {symbol} concluído e salvo no Supabase. Acurácia: {accuracy:.2f}.")
    return True, f"Modelo para {symbol} treinado com sucesso! Acurácia: {accuracy:.1%}"

async def get_prediction(symbol: str):
    try:
        response = supabase.table('trained_models').select("*").eq('symbol', symbol).execute()
        if not response.data:
            return None, "Modelo não encontrado. Por favor, treine o modelo primeiro usando o comando /treinar."
        
        row = response.data[0]
        
        model = pickle.loads(base64.b64decode(row['model_b64']))
        scaler = pickle.loads(base64.b64decode(row['scaler_b64']))
        metrics = json.loads(row['metrics_json'])
        metrics['last_trained'] = row['last_trained_utc']

    except Exception as e:
        logger.error(f"Erro ao carregar modelo do Supabase para {symbol}: {e}")
        return None, "Falha ao carregar o modelo treinado do banco de dados."

    df = await fetch_crypto_data(symbol=symbol, limit=250)
    if df is None or df.empty:
        return None, "Erro ao obter dados recentes para a previsão."

    df_features = create_features(df)
    if df_features.empty:
        return None, "Não foi possível gerar features com os dados recentes."

    features = ['open', 'high', 'low', 'volume', 'returns', 'log_returns', 'volatility', 
                'ma7', 'ma21', 'ma50', 'ma200' ,'rsi', 'atr', 'vwap', 'momentum']
    last_data = df_features[features].iloc[-1].to_frame().T
    last_data_scaled = scaler.transform(last_data)
    
    pred_class = model.predict(last_data_scaled)[0]
    pred_proba = model.predict_proba(last_data_scaled)[0]
    
    last_close = df_features['close'].iloc[-1]
    atr = df_features['atr'].iloc[-1]
    stop_loss, take_profit = calculate_risk_levels(last_close, atr)

    result = {
        "pred_class": pred_class, "pred_proba": pred_proba, "last_close": last_close,
        "stop_loss": stop_loss, "take_profit": take_profit, "metrics": metrics
    }
    return result, None

# --- Comandos do Bot do Telegram (sem alterações na lógica) ---

async def send_telegram_message(context: ContextTypes.DEFAULT_TYPE, chat_id, message):
    try:
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
    except TelegramError as e:
        logger.error(f"Erro ao enviar mensagem para o Telegram: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "👋 Bem-vindo ao Bot de Análise de Cripto!\n\n"
        "**Como usar:**\n"
        "1. Treine um modelo com: `/treinar ETH/USDT`\n"
        "2. Peça uma análise com: `/analisar ETH/USDT`\n\n"
        "Seus modelos agora são salvos na nuvem com o Supabase!"
    )
    await send_telegram_message(context, update.effective_chat.id, message)

async def treinar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await send_telegram_message(context, update.effective_chat.id, 
                                    "Por favor, especifique o par de moedas. Exemplo: `/treinar BTC/USDT`")
        return
    
    symbol = context.args[0].upper()
    await send_telegram_message(context, update.effective_chat.id, f"⏳ Iniciando treino para `{symbol}`. Os dados serão salvos na nuvem...")
    
    success, message = await train_and_save_model(symbol)
    
    await send_telegram_message(context, update.effective_chat.id, message)

async def analisar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await send_telegram_message(context, update.effective_chat.id, 
                                    "Por favor, especifique o par de moedas. Exemplo: `/analisar BTC/USDT`")
        return

    symbol = context.args[0].upper()
    await send_telegram_message(context, update.effective_chat.id, f"🔍 Buscando análise para `{symbol}` na nuvem...")

    result, error_message = await get_prediction(symbol)
    
    if error_message:
        await send_telegram_message(context, update.effective_chat.id, f"❌ {error_message}")
        return

    pred_class = result['pred_class']
    pred_proba = result['pred_proba']
    metrics = result['metrics']
    
    if pred_class == 1:
        direction = "ALTA 🟢"
        confidence = pred_proba[1]
    else:
        direction = "BAIXA 🔴"
        confidence = pred_proba[0]
    
    last_trained_str = pd.to_datetime(metrics['last_trained']).strftime('%Y-%m-%d %H:%M UTC')

    message = (
        f"📊 **Análise de Direção ({symbol})** 📊\n\n"
        f"*Último Preço:* `${result['last_close']:,.2f}`\n\n"
        f"▶️ *Previsão Próx. Período:* Tendência de *{direction}*\n"
        f"🤔 *Confiança do Modelo:* `{confidence:.1%}`\n\n"
        f"--- *Gestão de Risco* ---\n"
        f"*Stop-Loss Sugerido:* `${result['stop_loss']:,.2f}`\n"
        f"*Take-Profit Sugerido:* `${result['take_profit']:,.2f}`\n\n"
        f"--- *Métricas do Modelo (do último treino)* ---\n"
        f"*Acurácia:* `{metrics['accuracy']:.1%}`\n"
        f"*Precisão (ALTA):* `{metrics.get('precision_up', 0):.1%}`\n"
        f"*Precisão (BAIXA):* `{metrics.get('precision_down', 0):.1%}`\n"
        f"*Último Treino:* `{last_trained_str}`"
    )
    await send_telegram_message(context, update.effective_chat.id, message)

def main():
    if not all([SUPABASE_URL, SUPABASE_KEY, TELEGRAM_TOKEN]):
        logger.error("Variáveis de ambiente (SUPABASE_URL, SUPABASE_KEY, TELEGRAM_TOKEN) não foram configuradas. Verifique seu arquivo .env")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("treinar", treinar_command))
    application.add_handler(CommandHandler("analisar", analisar_command))
    
    logger.info("Bot iniciado. Usando Supabase para persistência na nuvem.")
    application.run_polling()

if __name__ == "__main__":
    main()
