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
   ##### Eigen
   cmake_opts=("-DCMAKE_BUILD_TYPE=${build_type}"
               "-DCMAKE_INSTALL_PREFIX=${install_dir}"
               "-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON")
   build_cmake_dep "eigen" ${cmake_opts[@]}

   eigen_include_dir="${install_dir}/include/eigen3"
   
   #################
   ##### Sophus
   cmake_opts=("-DCMAKE_BUILD_TYPE=${build_type}"
               "-DCMAKE_INSTALL_PREFIX=${install_dir}"
	       "-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON"
               "-DBUILD_TESTS=OFF"
               "-DEIGEN3_INCLUDE_DIR=${eigen_include_dir}")
   build_cmake_dep "Sophus" ${cmake_opts[@]}


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

