# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

unpack_file(){
    file=$1

    # echo "${file%.gz}"

    if gzip -t $file; then
        [[ -e ${file%.gz} ]] || gunzip -N "${file}"
    else 
        echo "Corrupt: $file"
    fi

}

N=16
open_sem $N

for file in `find blue_waters_dataset -mindepth 1 -type f -name "*.gz"`; do
    
    run_with_lock unpack_file $file
done 