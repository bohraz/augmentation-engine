#!/bin/bash

if [ $# -eq 0 ]; then
        echo "Usage: $0 [file/directory/special file paths...]"
        exit 1
fi

for ARG in "$@"; do
        if [ -f "$ARG" ]; then
                echo "$ARG is a regular file."
        elif [ -d "$ARG" ]; then
                echo "$ARG is a directory."
        elif [ -e "$ARG" ]; then
                echo "$ARG is a special file."
        else
                echo "Error: $ARG doesn't exist."
                exit 1
        fi
done
