startdate=2013-03-15
enddate=2013-04-14

curr="$startdate"
while true; do
    echo "$curr"
    [ "$curr" \< "$enddate" ] || break
    curr=$( date +%Y-%m-%d --date "$curr +1 day" )
done