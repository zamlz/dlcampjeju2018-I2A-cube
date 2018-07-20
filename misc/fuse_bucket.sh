#!/bin/sh

GS_BUCKET='experiments-rl'

gcsfuse ${GS_BUCKET} ./experiments
