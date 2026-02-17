import oscP5.*;
import netP5.*;

OscP5 oscP5;

// Variables to store received data
String[] fingers = new String[0];

float x = 0;
float y = 0;
float z = 0;
float velocity = 0;
float angle = 0;
PImage img;
PImage one;
PImage two;
PImage three;
PImage four;

boolean values = false;

void setup() {
  size(800, 600);
  
  frameRate(12);
  
  // Start oscP5, listening for incoming messages on port 5005
  oscP5 = new OscP5(this, 5007);
  img = loadImage("hand.png");
  one = loadImage("1.png");
  two = loadImage("2.png");
  three = loadImage("3.png");
  four = loadImage("4.png");
  
  println("Listening for OSC messages on port 5005");
}

void draw() {
  background(50);
  
  // Display the received data
  fill(255);
  if (values == true){
    textSize(12);
    text("Fingers: " + join(fingers, ","), 50, 50);
    text("X: " + nf(x, 1, 3), 50, 100);
    text("Y: " + nf(y, 1, 3), 50, 150);
    text("Z: " + nf(z, 1, 3), 50, 200);
    text("Velocity: " + nf(velocity, 1, 3), 50, 250);
    text("Angle: " + nf(angle, 1, 2), 50, 300);
  }
  
  imageMode(CENTER);
  pushMatrix();
  translate((1-x) * width,y * height);
  rotate(map(angle,0,1,-PI/2,PI/2));
  scale(map(z,0,1,0.3,1));
  if (fingers.length > 0){
    image(img, 0, 0);
    
    for (String f : fingers) {
      int val = int(f);
  
      if (val == 1) {
        image(one, 0, 0);
      } 
      else if (val == 2) {
        image(two, 0, 0);
      } 
      else if (val == 3) {
        image(three, 0, 0);
      } 
      else if (val == 4) {
        image(four, 0, 0);
      }
    }
  }
  popMatrix();
}

// This function is called automatically when OSC messages arrive
void oscEvent(OscMessage msg) {
  
  println(msg);
  
  // Check the address pattern and extract the value
  if (msg.checkAddrPattern("/fingers")) {
    String raw = msg.get(0).stringValue();
    String[] parts = raw.split(",");
  
    ArrayList<String> valid = new ArrayList<String>();
    for (String p : parts) {
      if (!p.trim().equals("")) {
        valid.add(p);
      }
    }
  
    fingers = valid.toArray(new String[0]);
  }
  else if (msg.checkAddrPattern("/x")) {
    x = msg.get(0).floatValue();
  }
  else if (msg.checkAddrPattern("/y")) {
    y = msg.get(0).floatValue();
  }
  else if (msg.checkAddrPattern("/z")) {
    z = msg.get(0).floatValue();
  }
  else if (msg.checkAddrPattern("/velocity")) {
    velocity = msg.get(0).floatValue();
  }
  else if (msg.checkAddrPattern("/angle")) {
    angle = msg.get(0).floatValue();
  }
}
