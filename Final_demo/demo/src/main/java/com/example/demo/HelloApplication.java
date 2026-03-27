package com.example.demo;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class HelloApplication extends Application {

    @Override
    public void start(Stage stage) {
        try {
            FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));
            Scene scene = new Scene(fxmlLoader.load(), 500, 600);
            stage.setTitle("Camera PDF Generator");
            stage.setScene(scene);
            stage.show();
        } catch (javafx.fxml.LoadException e) {
            System.err.println("Error loading FXML: " + e.getMessage());
        } catch (java.io.IOException e) {
            System.err.println("Error loading FXML: " + e.getMessage());
        }
    }


    public static void main(String[] args) {
        launch();
    }
}
