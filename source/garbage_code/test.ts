function displayTotalPerPerson(person: string, total: number) {
    let message: string = "Total for " + person + " is " + total;
    // @ts-ignore
    document.getElementById("totalMessage").innerText = message;
}