from typing import Literal

from pydantic import BaseModel, Field


class LoanInput(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: int = Field(..., ge=0)
    CoapplicantIncome: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=1)
    Loan_Amount_Term: float = Field(..., ge=1)
    Credit_History: float = Field(..., ge=0, le=1)
    Property_Area: Literal["Urban", "Rural", "Semiurban"]


class PredictionOutput(BaseModel):
    approval_probability: float
    prediction: int
    decision: str
