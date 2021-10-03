from flask import Blueprint, render_template, request
import pandas as pd
from readability_pred import CLRDataset, preprocessing, cal_read_o_time, cal_total_read_o_time, predict_text
bp = Blueprint('english', __name__)
