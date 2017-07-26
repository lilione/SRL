#!/bin/bash

function evaluate {
	src_file="$1".py
	python $src_file |& tee "result_$1".txt
}

function testment {
	evaluate CGNN
}

testment
