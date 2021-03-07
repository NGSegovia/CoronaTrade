import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.ml.neuralnet.MapUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.local.transforms.LocalTransformExecutor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

public class MainMSFT {

    private static final int FEATURES_COUNT = 4;
    private static final int CLASSES_COUNT = 3;

    public static void main(String[] args) throws Exception {
        // Set neural network parameters
        int NB_INPUTS = 86;
        int NB_EPOCHS = 10;
        int RANDOM_SEED = 1234;
        double LEARNING_RATE = 0.005;
        int BATCH_SIZE = 32;
        int LSTM_LAYER_SIZE = 200;
        int NUM_LABEL_CLASSES = 2;



        //=====================================================================
        //                 Step 1: Define the input data schema
        //=====================================================================

        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        Schema inputDataSchema = new Schema.Builder()
                //We can define a single column
                .addColumnString("Fecha")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Cerrar")
                //We can define different column types for different types of data:
                .addColumnInteger("Volumen")
                .addColumnsString("Abrir", "Alto", "Bajo")
                .build();


        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());


        Map<String, String> transform = Collections.singletonMap("\\$(.*)", "$1");


        //=====================================================================
        //            Step 2: Define the operations we want to do
        //=====================================================================

        //Lets define some operations to execute on the data...
        //We do this by defining a TransformProcess
        //At each step, we identify column by the name we gave them in the input data schema, above
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                //Let's remove some column we don't need
                .removeColumns("Cerrar","Abrir","Bajo")

                //Let's suppose our data source isn't perfect, and we have some invalid data: negative dollar amounts that we want to replace with 0.0
                //For positive dollar amounts, we don't want to modify those values
                //Use the ConditionalReplaceValueTransform on the "TransactionAmountUSD" column:
                .replaceStringTransform("Alto", transform)
                .convertToDouble("Alto")

                //Finally, let's suppose we want to parse our date/time column in a format like "2016/01/01 17:50.000"
                //We use JodaTime internally, so formats can be specified as follows: http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
                .stringToTimeTransform("Fecha", "MM/DD/YYYY", DateTimeZone.UTC)

                //We've finished with the sequence of operations we want to do: let's create the final TransformProcess object
                .build();

        //Now, print the schema after each time step:
        int numActions = tp.getActionList().size();

        for(int i=0; i<numActions; i++ ){
            System.out.println("\n\n==================================================");
            System.out.println("-- Schema after step " + i + " (" + tp.getActionList().get(i) + ") --");

            System.out.println(tp.getSchemaAfterStep(i));
        }


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


        //=====================================================================
        //      Step 3: Load our data and execute the operations locally
        //=====================================================================

        //Define input and output paths:
        File inputFile = new ClassPathResource("HistoricalData_1614354695394.csv").getFile(); //Normally just define your directory like "file:/..." or "hdfs:/..."
        File outputFile = new File("FilteredData.csv");
        if(outputFile.exists()){
            outputFile.delete();
        }
        outputFile.createNewFile();

        //Define input reader and output writer:
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(inputFile));

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);

        //Process the data:
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        rw.writeBatch(processedData);
        rw.close();


        //Print before + after:
        System.out.println("\n\n---- Original Data File ----");
        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);

        System.out.println("\n\nDONE");
    }


}
