import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import ccxt.async_support as ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
import asyncio
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Configuração de logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Função para coletar dados em tempo real
# A função já estava pronta para receber um símbolo, agora vamos usá-lo dinamicamente
async def fetch_crypto_data(symbol='BTC/USDT', timeframe='1h', limit=500):
    try:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })
        await exchange.load_markets()
        # Verifica se o símbolo existe na corretora
        if symbol not in exchange.markets:
            logger.error(f"Símbolo {symbol} não encontrado na Binance.")
            await exchange.close()
            return None
            
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        await exchange.close()
        return df
    except ccxt.BadSymbol as e:
        logger.error(f"Erro de Símbolo Inválido ao coletar dados para {symbol}: {e}")
        await exchange.close()
        return None
    except Exception as e:
        logger.error(f"Erro ao coletar dados para {symbol}: {e}")
        # Garante que a conexão seja fechada em caso de erro
        if 'exchange' in locals() and exchange.session:
            await exchange.close()
        return None

# Função para calcular ATR (Average True Range)
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Função para criar features
def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df)
    df = df.dropna()
    return df

# Função para calcular RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    if loss.eq(0).all():
        return pd.Series(100.0, index=series.index)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Função para calcular níveis de stop-loss e take-profit
def calculate_risk_levels(last_close, atr, risk_factor=1.5, reward_factor=2.0):
    stop_loss = last_close - (atr * risk_factor)
    take_profit = last_close + (atr * reward_factor)
    return stop_loss, take_profit

# Função para treinar o modelo e prever
async def train_and_predict(df):
    features = ['open', 'high', 'low', 'volume', 'returns', 'volatility', 'ma7', 'ma21', 'rsi', 'atr']
    X = df[features]
    y = df['close'].shift(-1)
    X = X[:-1]
    y = y[:-1]

    if len(X) < 10: # Garante que há dados suficientes para treinar
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    logger.info(f"Cross-Validation R2 Scores: {scores.mean():.2f} ± {scores.std():.2f}")
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    last_data = df[features].iloc[-1].to_frame().T
    last_data_scaled = scaler.transform(last_data)
    next_period_pred = model.predict(last_data_scaled)[0]
    
    last_close = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    stop_loss, take_profit = calculate_risk_levels(last_close, atr)
    
    return next_period_pred, mse, r2, last_close, stop_loss, take_profit

# Função para enviar mensagens via Telegram (simplificada)
async def send_telegram_message(context: ContextTypes.DEFAULT_TYPE, chat_id, message):
    try:
        await context.bot.send_message(chat_id=chat_id, text=message)
    except TelegramError as e:
        logger.error(f"Erro ao enviar mensagem para o Telegram: {e}")

# Comando /start <-- MUDANÇA: Mensagem inicial atualizada
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "👋 Bem-vindo ao Bot de Análise de Cripto!\n\n"
        "Use o comando /analisar seguido pelo par de moedas para obter uma análise.\n\n"
        "✅ **Exemplo:** `/analisar ETH/USDT`"
    )
    await send_telegram_message(context, update.effective_chat.id, message)

# Comando /analisar <-- MUDANÇA: Antigo /predict, agora dinâmico
async def analisar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Verifica se o usuário enviou um argumento com o comando
    if not context.args:
        await send_telegram_message(context, update.effective_chat.id, 
                                    "Por favor, especifique o par de moedas. Exemplo: /analisar BTC/USDT")
        return

    # Pega o primeiro argumento como o símbolo e converte para maiúsculas
    symbol = context.args[0].upper()
    await send_telegram_message(context, update.effective_chat.id, f"🔍 Analisando {symbol}... Por favor, aguarde.")

    # Busca os dados para o símbolo especificado
    df = await fetch_crypto_data(symbol=symbol)
    if df is None or df.empty:
        await send_telegram_message(context, update.effective_chat.id,
                                    f"❌ Erro ao coletar dados para {symbol}. Verifique se o par é válido e tente novamente.")
        return
    
    df = create_features(df)
    
    # Verifica se o dataframe tem dados suficientes após criar as features
    if df.empty:
        await send_telegram_message(context, update.effective_chat.id,
                                    f"❌ Não há dados suficientes para analisar {symbol} no momento.")
        return

    prediction_results = await train_and_predict(df)
    
    if prediction_results is None:
        await send_telegram_message(context, update.effective_chat.id,
                                    f"❌ Não foi possível treinar o modelo para {symbol} por falta de dados.")
        return

    next_period_pred, mse, r2, last_close, stop_loss, take_profit = prediction_results

    opportunity = "Manter posição"
    if next_period_pred > last_close * 1.01: # Threshold de 1% para compra
        opportunity = "Oportunidade de COMPRA 🟢"
    elif next_period_pred < last_close * 0.99: # Threshold de 1% para venda
        opportunity = "Oportunidade de VENDA 🔴"
    
    # <-- MUDANÇA: Mensagem de resposta agora usa o 'symbol' dinamicamente
    message = (
        f"📊 **Análise de Preço ({symbol})** 📊\n\n"
        f"Último Preço: ${last_close:,.2f}\n"
        f"Previsão Próx. Período: ${next_period_pred:,.2f}\n\n"
        f"Stop-Loss Sugerido: ${stop_loss:,.2f}\n"
        f"Take-Profit Sugerido: ${take_profit:,.2f}\n\n"
        f"**Oportunidade: {opportunity}**\n\n"
        f"🔍 Métricas do Modelo:\n"
        f"  - MSE: {mse:.4f}\n"
        f"  - R²: {r2:.2f}"
    )
    await send_telegram_message(context, update.effective_chat.id, message)

# Função principal
def main():
    # <-- MUDANÇA: Usa a nova classe Application Builder, que é o padrão atual da biblioteca
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    # <-- MUDANÇA: Registra o novo comando /analisar
    application.add_handler(CommandHandler("analisar", analisar))
    
    # Inicia o bot
    application.run_polling()

if __name__ == "__main__":
    main()