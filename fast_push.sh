#!/bin/sh

msg=$1

git add .
git commit -m $msg --author="DidierLeBail <didierm.lebail@gmail.com>"
git push origin aurelien
