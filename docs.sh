#!/bin/sh
run_mkdocs()
{
    poetry run mkdocs "$@"
}

prepare_api_docs()
{
    rm -rf docs/statistical_methods_library
    poetry run pdoc3 -o docs statistical_methods_library
}

serve_docs()
{
    prepare_api_docs
    run_mkdocs serve
}

build_docs()
{
    prepare_api_docs
    run_mkdocs build
}

usage="
$0 - build or view the Statistical Methods Library docs.

Available commands:
build - Build the docs placing the created site in $(dirname $0)/site
help - this message
serve - launch a local dev server to view the docs (building not required)
"

case "$1" in
build)
    build_docs
    ;;

help)
    echo "$usage"
    ;;

serve)
    serve_docs
    ;;

*)
    echo "Unknown command '$1' please use help for usage" 1>&2
    exit 1
    ;;
esac

