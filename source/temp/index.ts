export module SmokeTest {

    export function HelloWorld() {
        console.log("Hello World");
    }
}

class Logger {
    log(message:string):void {
        console.log(message);
    }

}

const logger = new Logger();
logger.log("hello world");