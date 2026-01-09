from datetime import datetime, timedelta
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.schemas import CandleSource
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("TOKEN")


def get_data_by_figi(figi: str, start, end) -> pd.DataFrame:
    data = []

    with Client(TOKEN) as c:
        for candle in c.get_all_candles(
            instrument_id=figi,
            from_=start,
            to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
            candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,
        ):
            data.append({
                'date': candle.time.date(),
                'price': candle.close.units + candle.close.nano / 1e9,
            })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df


def calculate(target: pd.DataFrame, index: pd.DataFrame) -> tuple[float, float]:
    general = target.merge(index, on='date', suffixes=("_target", "_index"))
    general["log_return_target"] = np.log(general["price_target"] / general["price_target"].shift())
    general["log_return_index"] = np.log(general["price_index"] / general["price_index"].shift())
    general.dropna(inplace=True)

    A = np.vstack([general["log_return_index"], np.ones(len(general["log_return_index"]))]).T
    b, a = np.linalg.lstsq(A, general["log_return_target"])[0]

    return round(b, 2), round(a, 3)


def main():
    index_figi = "BBG004730JJ5"
    target_figi = {
        "LKOH": ("BBG004731032", 0.62),
        "OZON": ("TCS80A10CW95", 0.44),
        "PLZL": ("BBG000R607Y3", 0.43),
        "SBER": ("BBG004730N88", 0.5),
        "YDEX": ("TCS00A107T19", 0.69),
        "T": ("TCS80A107UL4", 0.65),
        "SFIN": ("BBG003LYCMB1", 0.68),
        "VSMO": ("BBG004S68CV8", 1.02),
        "OZPH": ("TCS00A109B25", 0.51),
    }
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - relativedelta(months=12)
    index_df = get_data_by_figi(figi=index_figi, start=start_date, end=end_date)

    print("Label (diff. %) - QuantPiLL, T-broker")
    for label, data in target_figi.items():
        figi, broker_beta = data
        target_df = get_data_by_figi(figi, start=start_date, end=end_date)
        beta, alfa = calculate(target_df, index_df)
        diff = ((beta - broker_beta) / broker_beta) * 100
        print(f"{label} {diff:+.2f}% - {beta}, {broker_beta}")


if __name__ == "__main__":
    main()
