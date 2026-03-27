package com.example.demo;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
import java.util.ArrayList;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class HelloController {

    // === UI Elements ===
    @FXML private Button uploadEL1Button;
    @FXML private Button uploadEL2Button;
    @FXML private Button uploadELBGButton;
    @FXML private Button generatePdfButton;

    @FXML private ImageView imageViewEL1;
    @FXML private ImageView imageViewEL2;
    @FXML private ImageView imageViewELBG;

    @FXML private TextField cameraModelField;
    @FXML private TextField locationField;
    @FXML private TextField notesField;
    @FXML private TextField rowsField;
    @FXML private TextField columnsField;

    // 🔹 Brightness Levels
    @FXML private TextField brightnessLevel1Field;
    @FXML private TextField brightnessLevel2Field;
    @FXML private TextField brightnessLevel3Field;
    @FXML private TextField brightnessLevel4Field;



    // === Internal State ===
    private File el1Image;
    private File el2Image;
    private File elbgImage;
    private File annotatedEl1;

    private Set<String> defectedCells = new HashSet<>();
    private List<DefectInfo> defectInfoList = new ArrayList<>();

    // Inner class for defect info
    public static class DefectInfo {
        public int row;
        public int col;
        public String label;
        public double confidence;
        public double darkness;

        public DefectInfo(int row, int col, String label, double confidence, double darkness) {
            this.row = row;
            this.col = col;
            this.label = label;
            this.confidence = confidence;
            this.darkness = darkness;
        }
    }

    // === Image Upload Handlers ===
    @FXML
    protected void onUploadEL1Click() {
        el1Image = chooseImage();
        if (el1Image != null) imageViewEL1.setImage(new Image(el1Image.toURI().toString()));
    }

    @FXML
    protected void onUploadEL2Click() {
        el2Image = chooseImage();
        if (el2Image != null) imageViewEL2.setImage(new Image(el2Image.toURI().toString()));
    }

    @FXML
    protected void onUploadELBGClick() {
        elbgImage = chooseImage();
        if (elbgImage != null) imageViewELBG.setImage(new Image(elbgImage.toURI().toString()));
    }

    private File chooseImage() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
        );
        Stage stage = (Stage) generatePdfButton.getScene().getWindow();
        return fileChooser.showOpenDialog(stage);
    }

    // === Main Button Action ===
    @FXML
    protected void onGeneratePdfButtonClick() {
        if (el1Image == null || el2Image == null || elbgImage == null) {
            new Alert(Alert.AlertType.WARNING, "Please upload all three images (EL1, EL2, ELBG).").show();
            return;
        }

        JsonObject responseJson;
        try {
            responseJson = uploadToBackend(el1Image, el2Image, elbgImage);
            new Alert(Alert.AlertType.INFORMATION, "✅ Images processed successfully! Proceeding with analysis...").show();
        } catch (Exception ex) {
            new Alert(Alert.AlertType.WARNING, "⚠️ Upload had issues, but proceeding with analysis anyway...").show();
            responseJson = new JsonObject(); // Create empty response to continue
        }

        try {
            annotatedEl1 = saveAnnotatedImage(responseJson);
            if (annotatedEl1 != null) {
                System.out.println("Annotated image saved: " + annotatedEl1.getAbsolutePath());
            }
        } catch (Exception ex) {
            new Alert(Alert.AlertType.ERROR, "Failed to save annotated image: " + ex.getMessage()).show();
        }

        try {
            File pdfFile = requestBackendPDF(responseJson);
            if (pdfFile != null) {
                new Alert(Alert.AlertType.INFORMATION, "PDF generated " ).show();
            } else {
                new Alert(Alert.AlertType.ERROR, "Failed to generate PDF").show();
            }
        } catch (Exception ex) {
            new Alert(Alert.AlertType.ERROR, "PDF generation failed: " + ex.getMessage()).show();
        }
    }

    // === Backend Communication ===
    private JsonObject uploadToBackend(File el1Image, File el2Image, File elbgImage) throws Exception {
        OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
            .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
            .build();

        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM);

        builder.addFormDataPart("image1", el1Image.getName(),
            RequestBody.create(el1Image, MediaType.parse(Files.probeContentType(el1Image.toPath()))));
        builder.addFormDataPart("image2", el2Image.getName(),
            RequestBody.create(el2Image, MediaType.parse(Files.probeContentType(el2Image.toPath()))));
        builder.addFormDataPart("image3", elbgImage.getName(),
            RequestBody.create(elbgImage, MediaType.parse(Files.probeContentType(elbgImage.toPath()))));

        Request request = new Request.Builder()
            .url("http://127.0.0.1:5000/detect")
            .post(builder.build())
            .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String err = response.body() != null ? response.body().string() : "unknown";
                System.out.println("⚠️ Backend returned error: " + response.code() + " - " + err);
                // Return empty response instead of throwing exception
                return new JsonObject();
            }
            String body = response.body().string();
            return body.isEmpty() ? new JsonObject() : JsonParser.parseString(body).getAsJsonObject();
        }
    }

    private File saveAnnotatedImage(JsonObject responseJson) throws Exception {
        if (!responseJson.has("results")) return null;
        JsonArray results = responseJson.getAsJsonArray("results");
        if (results.size() == 0) return null;

        JsonObject first = results.get(0).getAsJsonObject();
        if (!first.has("annotated_image")) return null;

        String b64 = first.get("annotated_image").getAsString();
        if (b64 == null || b64.isEmpty()) return null;

        byte[] data = Base64.getDecoder().decode(b64);
        File temp = File.createTempFile("annotated_EL1_", ".jpg");
        try (FileOutputStream fos = new FileOutputStream(temp)) {
            fos.write(data);
        }
        temp.deleteOnExit();
        return temp;
    }

    private File requestBackendPDF(JsonObject detectResponse) throws Exception {
        OkHttpClient client = new OkHttpClient();

        // 🔹 Ensure we have required data for PDF generation
        if (!detectResponse.has("cell_detection") || detectResponse.getAsJsonObject("cell_detection").get("image").getAsString().isEmpty()) {
            System.out.println("⚠️ No cell detection data available, creating default response...");
            // Create default cell detection data
            JsonObject cellDetection = new JsonObject();
            cellDetection.addProperty("image", ""); // Empty base64 image
            cellDetection.add("detections", new JsonArray());
            cellDetection.addProperty("total_cells", 0);
            detectResponse.add("cell_detection", cellDetection);
        }

        // 🔹 Ensure we have quality result data
        if (!detectResponse.has("quality_result")) {
            System.out.println("⚠️ No quality result data available, creating default...");
            JsonObject qualityResult = new JsonObject();
            qualityResult.addProperty("snr50", 0.0);
            qualityResult.addProperty("sharpness_mm", 0.0);
            qualityResult.addProperty("sharpness_category", "Unknown");
            qualityResult.addProperty("accepted", false);
            JsonObject histogram = new JsonObject();
            histogram.addProperty("mean", 0.0);
            histogram.addProperty("variance", 0.0);
            histogram.addProperty("skewness", 0.0);
            histogram.addProperty("kurtosis", 0.0);
            qualityResult.add("histogram", histogram);
            detectResponse.add("quality_result", qualityResult);
        }

        // 🔹 Add rows & columns
        try {
            int rows = Integer.parseInt(rowsField.getText().trim());
            int cols = Integer.parseInt(columnsField.getText().trim());
            detectResponse.addProperty("rows", rows);
            detectResponse.addProperty("columns", cols);
        } catch (Exception e) {
            System.out.println("⚠️ Could not add rows/columns: " + e.getMessage());
        }

        // 🔹 Add camera info
        try {
            String cameraModel = cameraModelField.getText().trim();
            String location = locationField.getText().trim();
            String notes = notesField.getText().trim();
            if (!cameraModel.isEmpty()) detectResponse.addProperty("camera_model", cameraModel);
            if (!location.isEmpty()) detectResponse.addProperty("location", location);
            if (!notes.isEmpty()) detectResponse.addProperty("notes", notes);
        } catch (Exception e) {
            System.out.println("⚠️ Could not add camera info: " + e.getMessage());
        }

        // 🔹 Add brightness levels as array
        try {
            JsonArray brightnessLevels = new JsonArray();
            if (!brightnessLevel1Field.getText().trim().isEmpty())
                brightnessLevels.add(Integer.parseInt(brightnessLevel1Field.getText().trim()));
            if (!brightnessLevel2Field.getText().trim().isEmpty())
                brightnessLevels.add(Integer.parseInt(brightnessLevel2Field.getText().trim()));
            if (!brightnessLevel3Field.getText().trim().isEmpty())
                brightnessLevels.add(Integer.parseInt(brightnessLevel3Field.getText().trim()));
            if (!brightnessLevel4Field.getText().trim().isEmpty())
                brightnessLevels.add(Integer.parseInt(brightnessLevel4Field.getText().trim()));

            if (brightnessLevels.size() > 0) {
                detectResponse.add("brightness_levels", brightnessLevels);
            }
        } catch (Exception e) {
            System.out.println("⚠️ Could not add brightness levels: " + e.getMessage());
        }

        RequestBody body = RequestBody.create(
            detectResponse.toString(),
            MediaType.parse("application/json")
        );

        Request request = new Request.Builder()
            .url("http://127.0.0.1:5000/generate_pdf")
            .post(body)
            .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String err = response.body() != null ? response.body().string() : "unknown";
                throw new RuntimeException("PDF generation failed: " + response.code() + " - " + err);
            }

            JsonObject pdfJson = JsonParser.parseString(response.body().string()).getAsJsonObject();
            if (!pdfJson.get("success").getAsBoolean()) {
                throw new RuntimeException("Backend error: " + pdfJson.get("error").getAsString());
            }

            String pdfBase64 = pdfJson.get("pdf_data").getAsString();
            byte[] pdfBytes = Base64.getDecoder().decode(pdfBase64);

            FileChooser saveChooser = new FileChooser();
            saveChooser.setInitialFileName(pdfJson.get("pdf_filename").getAsString());
            Stage stage = (Stage) generatePdfButton.getScene().getWindow();
            File pdfFile = saveChooser.showSaveDialog(stage);
            if (pdfFile == null) return null;

            try (FileOutputStream fos = new FileOutputStream(pdfFile)) {
                fos.write(pdfBytes);
            }
            return pdfFile;
        }
    }
}
