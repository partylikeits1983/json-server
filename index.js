// crypto 
const ETH = require('/home/ubuntu/Desktop/ETH.json');
const BTC = require('/home/ubuntu/Desktop/BTC.json');
const XMR = require('/home/ubuntu/Desktop/XMR.json');
const BNB = require('/home/ubuntu/Desktop/BNB.json');
const DOGE = require('/home/ubuntu/Desktop/DOGE.json');
const XRP = require('/home/ubuntu/Desktop/XRP.json');
const LTC = require('/home/ubuntu/Desktop/LTC.json');
const LINK = require('/home/ubuntu/Desktop/LINK.json');

//currency
const DXY = require('/home/ubuntu/Desktop/DXY.json');
const EUR = require('/home/ubuntu/Desktop/EUR.json');
const JPY = require('/home/ubuntu/Desktop/JPY.json');
const GBP = require('/home/ubuntu/Desktop/GBP.json');
const CHF = require('/home/ubuntu/Desktop/CHF.json');
const AUD = require('/home/ubuntu/Desktop/AUD.json');
const RUB = require('/home/ubuntu/Desktop/RUB.json');

// OHLC
const DXYOHLC = require('/home/ubuntu/Desktop/DXYOHLC.json');
const EUROHLC = require('/home/ubuntu/Desktop/EUROHLC.json');
const JPYOHLC = require('/home/ubuntu/Desktop/JPYOHLC.json');
const GBPOHLC = require('/home/ubuntu/Desktop/GBPOHLC.json');
const CHFOHLC = require('/home/ubuntu/Desktop/CHFOHLC.json');
const AUDOHLC = require('/home/ubuntu/Desktop/AUDOHLC.json');
const RUBOHLC = require('/home/ubuntu/Desktop/RUBOHLC.json');

// yield curve data
const YIELD = require('/home/ubuntu/Desktop/yield.json');

module.exports = () => ({
  ETH: ETH,
  BTC: BTC,
  XMR: XMR,
  BNB: BNB,
  DOGE: DOGE,
  XRP: XRP,
  LTC: LTC,
  LINK: LINK,

//currency
  DXY: DXY,
  EUR: EUR,
  JPY: JPY,
  GBP: GBP,
  CHF: CHF,
  AUD: AUD,
  RUB: RUB,

// OHLC
  DXYOHLC: DXYOHLC,
  EUROHLC: EUROHLC,
  JPYOHLC: JPYOHLC,
  GBPOHLC: GBPOHLC,
  CHFOHLC: CHFOHLC,
  AUDOHLC: AUDOHLC,
  RUBOHLC: RUBOHLC,
  
// yield curves
  YIELD: YIELD,

});


