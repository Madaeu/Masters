#!/bin/bash

build_type='Release'
: ${CMAKE:=cmake}

num_threads=1

function build_cmake_dep()
{
   local depdir=$1
   shift
   local cmakeopts=("$@")

   echo "Building ${depdir})"

   builddir=${build_dir}/${depdir}
   ((rebuild)) && ${CMAKE} -E remove_directory ${builddir}
   if [[ ! -e ${builddir} ]]; then
     ${CMAKE} -E make_directory ${builddir} || exit
     ${CMAKE} -E chdir ${builddir} ${CMAKE} "${cmakeopts[@]}" $(realpath ${depdir}) || exit
   fi

   ${CMAKE} --build ${builddir} --target install -- -j${num_threads} || exit
}

function build_cmake_deps()
{
   build_dir="$(pwd)/build"
   install_dir="$(pwd)/install"
   

   #################
   ##### DBoW2
   cmake_opts=("-DCMAKE_BUILD_TYPE=${build_type}"
               "-DCMAKE_INSTALL_PREFIX=${install_dir}")
   build_cmake_dep "DBoW2" ${cmake_opts[@]}

}

function usage()
{
   echo "Usage: $0 [--threads <num_threads>] [--rebuild]"
}

function parse_args()
{
  options=$(getopt -o dv --long help --long rebuild --long threads: "$@")
  [ $? -eq 0 ] || {
    usage
    exit 1
  }
  eval set -- "$options"
  while true; do
    case "$1" in
      --threads)
        shift
        num_threads=$1
        ;;
      --rebuild)
        rebuild=1
        ;;
      --help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done
}


parse_args $0 "$@"

pushd $(dirname $0)
build_cmake_deps
popd

