"""Text context for the LLM text-context experiment. ONLY information known at the forecast origin:
a description of the series (what it is, region/page/BA identity, its usual seasonal pattern as
general domain knowledge) and the calendar of the forecast window (dates, US federal holidays).
No values from the window, no news, no weather. Same wording for every item of a task.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

_CAL = USFederalHolidayCalendar()

HHS = {"nat": "the United States (national)",
       "hhs1": "HHS Region 1 (CT, ME, MA, NH, RI, VT)", "hhs2": "HHS Region 2 (NJ, NY, PR, VI)",
       "hhs3": "HHS Region 3 (DE, DC, MD, PA, VA, WV)", "hhs4": "HHS Region 4 (AL, FL, GA, KY, MS, NC, SC, TN)",
       "hhs5": "HHS Region 5 (IL, IN, MI, MN, OH, WI)", "hhs6": "HHS Region 6 (AR, LA, NM, OK, TX)",
       "hhs7": "HHS Region 7 (IA, KS, MO, NE)", "hhs8": "HHS Region 8 (CO, MT, ND, SD, UT, WY)",
       "hhs9": "HHS Region 9 (AZ, CA, HI, NV)", "hhs10": "HHS Region 10 (AK, ID, OR, WA)"}


def _holidays(first, last) -> str:
    h = _CAL.holidays(start=pd.Timestamp(first) - pd.Timedelta(days=3),
                      end=pd.Timestamp(last) + pd.Timedelta(days=3), return_name=True)
    inside = [f"{d:%Y-%m-%d} ({n})" for d, n in h.items() if first <= d <= last]
    return ", ".join(inside) if inside else "none"


def context_text(task: str, uid: str, future_ds: list) -> str:
    first, last = pd.Timestamp(future_ds[0]), pd.Timestamp(future_ds[-1])
    window = f"Forecast window: {first:%Y-%m-%d} to {last:%Y-%m-%d}. US federal holidays in the window: {_holidays(first, last)}."
    if task == "flu":
        desc = (f"Series: weekly percentage of outpatient visits for influenza-like illness (ILI) in {HHS.get(uid, uid)}, "
                "as first reported (later revised). US influenza activity is usually highest from December to February "
                "and lowest in summer.")
    elif task == "fred":
        desc = (f"Series: US macroeconomic indicator with FRED-MD mnemonic '{uid}', monthly "
                "(a real-time vintage, so recent values may be revised).")
    elif task == "wiki":
        desc = f"Series: daily page views of the English Wikipedia article '{uid.replace('_', ' ')}' (log scale)."
    elif task == "eia":
        desc = (f"Series: hourly electricity demand (megawatts) of the US balancing authority '{uid}'. "
                "Demand follows daily and weekly cycles and rises with heating and cooling needs.")
    else:
        desc = f"Series id: {uid}."
    return desc + " " + window
