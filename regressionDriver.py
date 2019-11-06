#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from regression import LinearRegression, LogisticRegression
import sys
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    argparse.ArgumentParser(description="Generate a Sequence Vector Machine (SVM).")
    parser.add_argument("type", type=str, choices=['linear', 'logistic'], help="Set a constant C for the SVM")
    parser.add_argument("--noPlot", help="Do not open a graphical representation of the SVM.", action="store_true")
    parser.add_argument("--loo", help="Calculate leave-one-out error. Will have an adverse effect on run-time.", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    settings = dict()

    if args.noPlot:
        settings['plot'] = False
    else:
        settings['plot'] = True

    if args.loo:
        settings['loo'] = True
    else:
        settings['loo'] = False

    if args.type == 'linear':
        # Linear Regression
        LinearRegression(calculate_error=settings.get('loo'), plot_now=settings.get('plot'))
    elif args.type =='logistic':
        # Logistic Regression
        LogisticRegression(calculate_error=settings.get('loo'), plot_now=settings.get('plot'))