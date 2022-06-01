root_dir=`pwd`
for scene in `ls`
do
	if [ -d ${scene} ];then
		echo ${scene}
		for trajectory in `ls ${scene}`
		do
			if [ -d ${scene}/${trajectory} ]; then
				echo ${trajectory}
				output_dir=${scene}/${trajectory}/depth_fill
				if [ ! -d ${output_dir} ];then
					mkdir ${output_dir}
					echo ${output_dir}
				fi
				python depth_fill.py ${root_dir}/${scene}/${trajectory}
			fi
		done
	fi
done
