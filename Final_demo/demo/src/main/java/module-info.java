module com.example.demo {
    requires transitive javafx.controls;
    requires transitive javafx.fxml;
    requires transitive javafx.graphics;
    requires javafx.web;
    requires okhttp3;
    requires com.google.gson;
    requires org.apache.pdfbox;

    // Allow JavaFX to access HelloApplication
    opens com.example.demo to javafx.graphics, javafx.fxml;
    exports com.example.demo;
}
