CFLAGS = -I../inc -L../lib -Wl,-rpath='../lib' -g -lgm

all:test_md_live test_md_query test_md_playback test_td_live test_strategy

test_md_live:test_md_live.cpp
	g++  test_md_live.cpp -o test_md_live  ${CFLAGS}
test_md_query:test_md_query.cpp
	g++ test_md_query.cpp -o test_md_query ${CFLAGS}
test_md_playback:test_md_playback.cpp
	g++ test_md_playback.cpp -o test_md_playback ${CFLAGS}
test_td_live:test_td_live.cpp
	g++ test_td_live.cpp -o test_td_live ${CFLAGS}
test_strategy:test_strategy.cpp
	g++ test_strategy.cpp -o test_strategy ${CFLAGS}

clean:
	rm test_md_live test_md_query test_md_playback test_td_live test_strategy
