from flask import Flask, render_template, request

app = Flask(__name__)
query = None

def start_GUI(basic_query):

    """starts the server, allows user to input their query and get results"""
    @app.route("/", methods=["GET", "POST"])
    def search():
        if request.method == "POST":
            user_query = request.form["query"]
            urls = basic_query.query_index(user_query)
            return render_template("results.html", user_query=user_query, urls=urls)
        return render_template("search_engine.html")

    app.run(debug=True)