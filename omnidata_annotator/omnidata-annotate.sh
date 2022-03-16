#!/usr/bin/env bash


# Passed arguments : --model_path --task  with  {args}*

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path=*) 
          model_path="${1#*=}"
          ;; 
        --task=*) 
          task="${1#*=}"
          ;;
        with) 
        break
        ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ $# -ge 1 ]; then
  args="${@:2}"
else
  args=""
fi

# List of all tasks
all_tasks=(normal depth_zbuffer depth_euclidean reshading edge2d edge3d keypoints2d keypoints3d segment2d segment25d semantic)

case $task in
all)
  tasks=("${all_tasks[@]}")
  ;;
*)
  tasks=($task)
  ;;
esac

# Execute tasks
for task in "${tasks[@]}"; do
  echo "___________ Task : $task ___________"
  bash jobs/run_single_job.sh $task $model_path $args
done
