#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Book round robin shard: every engine = bin/G + bundle E options + W512_sf net, differing only in the book.
# usage: sbatch books/rr.sh STAGE SHARD ROUNDS ENGINES_FILE     (ENGINES_FILE: lines "name opt=val opt=val ...")
# Games start from books/seeds4ply.pgn (200 most common 4-ply lines of 2000+ Lichess games, random order per shard).
set -u
X=/scratch/ralbe/chess_nnue/exp2; FC=/scratch/ralbe/chess_nnue/tools/fastchess/fastchess
STAGE=$1; SHARD=$2; ROUNDS=$3; EF=$4
T=$X/books/rr/$STAGE; mkdir -p $T; cp -n $X/bin/G $T/G 2>/dev/null || true
E="$(sed "s#EvalFile=[^ ]*#EvalFile=$X/nets/W512_sf_s2/W512_sf_s2.nnue#" $X/ledger/E_opts.txt)"
args=()
while read -r name opts; do
  [ -z "$name" ] && continue
  a=(-engine cmd=$T/G name=$name)
  for kv in $E $opts; do a+=("option.${kv%%=*}=${kv#*=}"); done
  args+=("${a[@]}")
done < $EF
echo "stage $STAGE shard $SHARD rounds $ROUNDS engines $(awk '{print $1}' $EF | tr '\n' ' ')"; hostname
$FC "${args[@]}" -each tc=${TC:-6+0.06} option.Threads=1 option.Hash=32 -tournament roundrobin \
  -rounds $ROUNDS -games 2 -repeat -openings file=$X/books/seeds4ply.pgn format=pgn order=random -srand $((1000*SHARD+7)) \
  -concurrency 11 -recover -ratinginterval 0 -pgnout file=$T/shard$SHARD.pgn -config outname=$T/config$SHARD.json \
  | grep --line-buffered -vE "^(Warning|Info|Position|Moves);|^Started game|^Finished game"
echo "== done"
