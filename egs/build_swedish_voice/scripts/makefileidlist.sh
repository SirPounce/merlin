#!/bin/bash
#rm $2
ls $1 | sed 's/\(.*\)\..*/\1/' > $2
