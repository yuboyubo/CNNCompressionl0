package com.example.cnncompression;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewDebug;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Tensor;
import org.w3c.dom.Text;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {
    private Module module = null;
    private Scanner scanner = null;

    public String findDataset(String dataset_id) {
        String dataset = "";
        switch (dataset_id) {
            case "WISDM dataset":
                dataset = "wisdm";
                break;
            case "UCI-HAR dataset":
                dataset = "ucihar";
                break;
            case "PAMAP2 dataset":
                dataset = "pamap";
                break;
        }
        return dataset;
    }

    public String findModel(String penalty_id, String compress_id) {
        String model = "";
        if (compress_id.equals("Uncompressed Model")) {
            switch (penalty_id) {
                case "l0 norm":
                    model = "l0_norm_uncompressed.ptl";
                    break;
                case "l1 norm":
                    model = "l1_norm_uncompressed.ptl";
                    break;
                case "l2 norm":
                    model = "l2_norm_uncompressed.ptl";
                    break;
                case "group lasso":
                    model = "group_lasso_uncompressed.ptl";
                    break;
                case "l1 group lasso":
                    model = "l1_group_lasso_uncompressed.ptl";
                    break;
                case "l0 group lasso":
                    model = "l0_group_lasso_uncompressed.ptl";
                    break;
            }
        }
        else {
            switch (penalty_id) {
                case "l0 norm":
                    model = "l0_norm_compressed.ptl";
                    break;
                case "l1 norm":
                    model = "l1_norm_compressed.ptl";
                    break;
                case "l2 norm":
                    model = "l2_norm_compressed.ptl";
                    break;
                case "group lasso":
                    model = "group_lasso_compressed.ptl";
                    break;
                case "l1 group lasso":
                    model = "l1_group_lasso_compressed.ptl";
                    break;
                case "l0 group lasso":
                    model = "l0_group_lasso_compressed.ptl";
                    break;
            }
        }
        return model;
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public double calculateTime(String model, String dataset, int times) {
        int columns = 0;
        int rows = 0;
        int total_input = 200;
        if (dataset.equals("pamap")) {
            rows = 128;
            columns = 40;
        }
        else if (dataset.equals("wisdm")) {
            rows = 80;
            columns = 3;
        }
        else if (dataset.equals("ucihar")) {
            rows = 128;
            columns = 9;
        }

        double[][][] myArray = new double[rows][columns][total_input];
        try {
            for (int i = 0; i < total_input; i++) {
                scanner = new Scanner(new BufferedReader(new InputStreamReader(this.getAssets().open(dataset + "/test" +  String.valueOf(i)))));
                while (scanner.hasNextLine()) {
                    for (int j = 0; j < columns; j++) {
                        String[] line = scanner.nextLine().trim().split(",");
                        for (int k = 0; k < rows; k++) {
                            myArray[k][j][i] = Double.parseDouble(line[j]);
                        }
                    }
                }
                scanner.close();
            }
        }
        catch (IOException e) {
            Log.e("No such path","Error reading assets", e);
            finish();
        }

        DoubleBuffer dataBuf = ByteBuffer.allocateDirect(rows * columns * total_input * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
        for (int k = 0; k < total_input; k++) {
            for (int i = 0; i < columns; ++i) {
                for (int j = 0; j < rows; ++j) {
                    dataBuf.put(myArray[j][i][k]);
                }
            }
        }
        dataBuf.order();
        long start_time = 0;
        long end_time = 0;
        final Tensor inputTensor = Tensor.fromBlob(dataBuf,  new long[]{total_input, columns,rows});
        try {
            start_time = System.nanoTime();
            for (int i = 0; i < times; i++) {
                module = LiteModuleLoader.load(assetFilePath(this, dataset + "_" + model));
                module.forward(IValue.from(inputTensor)).toTensor();
            }
            end_time = System.nanoTime();
        } catch (IOException e) {
            Log.e("Error reading", "Error reading assets", e);
            finish();
        }
        return (end_time - start_time) / 1000000000.0;
    }

    public double runTest(String dataset_id, String penalty_id, String battery_id, String compress_id) {
        String model = "";
        String dataset = "";
        double time = 0;
        if (battery_id.equals("Running Time Test")) {
            model = findModel(penalty_id, compress_id);
            dataset = findDataset(dataset_id);
            time = calculateTime(model, dataset, 1);
        }
        else {
            model = findModel(penalty_id, compress_id);
            dataset = findDataset(dataset_id);
            time = calculateTime(model, dataset, 100);
        }
        return time;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        RadioGroup dataset = findViewById(R.id.dataset);
        RadioGroup penalty = findViewById(R.id.penalty);
        RadioGroup isbattery = findViewById(R.id.isbattery);
        RadioGroup iscompress = findViewById(R.id.iscompress);

        Button start_button = findViewById(R.id.start_button);
        TextView result_text = findViewById(R.id.result_veiw);

        start_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int selectedId = dataset.getCheckedRadioButtonId();
                int penaltyId = penalty.getCheckedRadioButtonId();
                int batteryId = isbattery.getCheckedRadioButtonId();
                int compressId = iscompress.getCheckedRadioButtonId();
                RadioButton dataset_button = findViewById(selectedId);
                RadioButton penalty_button = findViewById(penaltyId);
                RadioButton battery_button = findViewById(batteryId);
                RadioButton compress_button = findViewById(compressId);

                double running_time = runTest(dataset_button.getText().toString(), penalty_button.getText().toString(), battery_button.getText().toString(), compress_button.getText().toString());

                result_text.setText("Total running Time: " + String. format("%.2f", running_time) + " seconds");
            }
        });
    }
}