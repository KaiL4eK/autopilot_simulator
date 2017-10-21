import timeit
import textwrap



count = 10000

t1 = timeit.Timer(
    textwrap.dedent(
        """
        get_distance_to_4(np.array([1, 4], dtype=np.float32), np.array([5, 1], dtype=np.float32))
        """),
    setup_statement
)

print(t1.timeit(number=count))


t2 = timeit.Timer(
    textwrap.dedent(
        """
        get_distance_to(Point(1, 4), Point(5, 1))
        """),
    setup_statement
)

print(t2.timeit(number=count))

t2 = timeit.Timer(
    textwrap.dedent(
        """
        np.linalg.norm(np.array([1, 4]) - np.array([5, 1]))
        """),
    setup_statement
)

print(t2.timeit(number=count))	

