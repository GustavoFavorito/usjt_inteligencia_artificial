import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Knn {
    // xs.size() == ys.size()
    private static double sqDist(List<Double> xs, List<Double> ys) {
        Function<Double, Double> sq = x -> x * x;
        return IntStream.range(0, xs.size()).mapToDouble(   i -> (sq.apply(xs.get(i) - ys.get(i)))).sum();
    };

    public static String knn(int k, List<Double> sample, List<Pair<String, List<Double>>> trainingData,
                             BiFunction<List<Double>, List<Double>, Double> dist) {
        return trainingData.stream().map(p -> new Pair<>(p.fst, dist.apply(sample, p.snd)))
                    .sorted(Comparator.comparing(p -> p.snd))
                    .limit(k)
                    .collect(Collectors.groupingBy(p -> p.fst))
                    .entrySet()
                    .stream()
                    .sorted(Comparator.comparing(e -> e.getValue().size()))
                    .collect(Collectors.toList())
                    .get(0)
                    .getKey();
    };

    private static Pair<String, List<Double>> process(String line) {
        String[] vs = line.split(",");
        List<Double> rep = Stream.of(vs)
                .limit(4)
                .map(Double::parseDouble)
                .collect(Collectors.toList());
        return new Pair<>(vs[4], rep);
    };

    public static void main(String[] args) {
        try (Stream<String> file = Files.lines(Paths.get("C:\\Users\\Estudos\\Documents\\workspace\\inteligencia_artificial\\aula_01\\static\\iris.txt"))) {
            List<Pair<String, List<Double>>>
                    trainingData = file.map(line -> process(line))
                                        .collect(Collectors.toList());
            trainingData.forEach(
                    p -> System.out.println(p.fst + " : " + knn(7, p.snd, trainingData, Knn::sqDist)));
        } catch (IOException e) {
            e.printStackTrace();
        }
    };
};
