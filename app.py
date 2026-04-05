import os
import json
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, session
)
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from models import db, User, Assignment, Submission
from grader_queue.grading_queue import start_grading

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(Config.BASE_DIR, "instance"),    exist_ok=True)
os.makedirs(Config.DATASETS_FOLDER,                       exist_ok=True)
os.makedirs(Config.SUBMISSIONS_FOLDER,                    exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_notebook(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_NOTEBOOK_EXT

def allowed_dataset(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_DATASET_EXT

def _redirect_by_role(user):
    if user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    if user.role == "teacher":
        return redirect(url_for("teacher_dashboard"))
    return redirect(url_for("student_dashboard"))


# ══════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def login():
    # Always show the login form regardless of current session
    # Submitting it logs out the old user and logs in the new one
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash("Invalid email or password.", "danger")
            return render_template("auth/login.html")

        if user.role == "teacher" and user.status == "pending":
            flash("Your account is pending admin approval.", "warning")
            return render_template("auth/login.html")

        if user.role == "teacher" and user.status == "rejected":
            flash("Your registration was rejected. Contact admin.", "danger")
            return render_template("auth/login.html")

        # FIX 1: Force logout first, clear session, then login new user
        logout_user()
        session.clear()
        login_user(user, remember=False)
        return _redirect_by_role(user)

    # GET — always show the login form, even if someone is logged in
    return render_template("auth/login.html")


@app.route("/register/student", methods=["GET", "POST"])
def register_student():
    if request.method == "POST":
        email         = request.form.get("email", "").strip().lower()
        enrollment_no = request.form.get("enrollment_no", "").strip()

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register_student"))
        if User.query.filter_by(enrollment_no=enrollment_no).first():
            flash("Enrollment number already registered.", "danger")
            return redirect(url_for("register_student"))

        new_user = User(
            name          = request.form.get("name", "").strip(),
            email         = email,
            password      = generate_password_hash(
                                request.form.get("password"),
                                method="pbkdf2:sha256"),
            role          = "student",
            status        = "approved",
            enrollment_no = enrollment_no,
            branch        = request.form.get("branch", "").strip(),
            batch         = request.form.get("batch",  "").strip(),
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("auth/register.html", role="student")


@app.route("/register/teacher", methods=["GET", "POST"])
def register_teacher():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register_teacher"))

        new_user = User(
            name     = request.form.get("name", "").strip(),
            email    = email,
            password = generate_password_hash(
                           request.form.get("password"),
                           method="pbkdf2:sha256"),
            role     = "teacher",
            status   = "pending",
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration submitted! Await admin approval.", "info")
        return redirect(url_for("login"))

    return render_template("auth/register.html", role="teacher")


@app.route("/logout")
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════
#  ADMIN
# ══════════════════════════════════════════════════════════

@app.route("/admin")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        return "Access denied", 403
    pending_teachers  = User.query.filter_by(role="teacher", status="pending").all()
    approved_teachers = User.query.filter_by(role="teacher", status="approved").all()
    all_assignments   = Assignment.query.order_by(Assignment.created_at.desc()).all()
    return render_template(
        "admin/dashboard.html",
        pending_teachers  = pending_teachers,
        approved_teachers = approved_teachers,
        all_assignments   = all_assignments,
    )


@app.route("/admin/approve/<int:user_id>", methods=["POST"])
@login_required
def approve_teacher(user_id):
    if current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    user = User.query.get_or_404(user_id)
    user.status = "approved"
    db.session.commit()
    flash(f"{user.name} approved successfully.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/reject/<int:user_id>", methods=["POST"])
@login_required
def reject_teacher(user_id):
    if current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    user = User.query.get_or_404(user_id)
    user.status = "rejected"
    db.session.commit()
    flash(f"{user.name} rejected.", "warning")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/view_teacher/<int:teacher_id>")
@login_required
def admin_view_teacher(teacher_id):
    if current_user.role != "admin":
        return "Access denied", 403
    teacher     = User.query.get_or_404(teacher_id)
    assignments = Assignment.query.filter_by(teacher_id=teacher_id).all()
    students    = {u.id: u for u in User.query.filter_by(role="student").all()}
    return render_template(
        "admin/teacher_view.html",
        teacher     = teacher,
        assignments = assignments,
        students    = students,
    )


# ══════════════════════════════════════════════════════════
#  TEACHER
# ══════════════════════════════════════════════════════════

@app.route("/teacher")
@login_required
def teacher_dashboard():
    if current_user.role != "teacher":
        return "Access denied", 403
    assignments = Assignment.query.filter_by(
        teacher_id=current_user.id
    ).order_by(Assignment.created_at.desc()).all()
    students = {u.id: u for u in User.query.filter_by(role="student").all()}
    return render_template(
        "teacher/dashboard.html",
        assignments = assignments,
        students    = students,
    )


@app.route("/teacher/create_assignment", methods=["GET", "POST"])
@login_required
def create_assignment():
    if current_user.role != "teacher":
        return "Access denied", 403

    if request.method == "POST":
        title       = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        due_date    = request.form.get("due_date", "").strip()
        task_type   = request.form.get("task_type", "Custom").strip()
        rubric_json = request.form.get("rubric_json", "{}").strip()

        try:
            rubric_dict = json.loads(rubric_json)
            if not rubric_dict:
                flash("Rubric cannot be empty.", "danger")
                return redirect(url_for("create_assignment"))
        except json.JSONDecodeError:
            flash("Invalid rubric format.", "danger")
            return redirect(url_for("create_assignment"))

        dataset_path = None
        if "dataset" in request.files:
            file = request.files["dataset"]
            if file and file.filename and allowed_dataset(file.filename):
                filename     = secure_filename(file.filename)
                dataset_path = os.path.join(Config.DATASETS_FOLDER, filename)
                file.save(dataset_path)

        assignment = Assignment(
            title        = title,
            description  = description,
            due_date     = due_date,
            task_type    = task_type,
            rubric_json  = json.dumps(rubric_dict),
            dataset_path = dataset_path,
            teacher_id   = current_user.id,
        )
        db.session.add(assignment)
        db.session.commit()
        flash("Assignment created successfully!", "success")
        return redirect(url_for("teacher_dashboard"))

    return render_template("teacher/create_assignment.html")


@app.route("/teacher/assignment/<int:assign_id>")
@login_required
def teacher_view_assignment(assign_id):
    if current_user.role != "teacher":
        return "Access denied", 403
    assignment = Assignment.query.get_or_404(assign_id)
    if assignment.teacher_id != current_user.id:
        return "Access denied", 403
    submissions = Submission.query.filter_by(assignment_id=assign_id).all()
    students    = {u.id: u for u in User.query.filter_by(role="student").all()}
    return render_template(
        "teacher/view_results.html",
        assignment  = assignment,
        submissions = submissions,
        students    = students,
    )


@app.route("/teacher/grade_all/<int:assign_id>", methods=["POST"])
@login_required
def grade_all(assign_id):
    if current_user.role != "teacher":
        return jsonify({"error": "Unauthorized"}), 403
    assignment = Assignment.query.get_or_404(assign_id)
    if assignment.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    if assignment.grading_status == "grading":
        return jsonify({"success": False, "message": "Already grading."})

    gradeable = Submission.query.filter_by(
        assignment_id = assign_id,
        status        = "submitted"
    ).count()

    if gradeable == 0:
        return jsonify({
            "success": False,
            "message": "No new or resubmitted work to grade."
        })

    start_grading(app, assign_id)
    return jsonify({
        "success": True,
        "message": f"Grading started for {gradeable} submission(s).",
    })


# FIX 2: Detailed grading status — returns per-submission progress
@app.route("/teacher/grading_status/<int:assign_id>")
@login_required
def grading_status(assign_id):
    assignment  = Assignment.query.get_or_404(assign_id)
    submissions = Submission.query.filter_by(assignment_id=assign_id).all()
    students    = {u.id: u for u in User.query.filter_by(role="student").all()}

    breakdown = []
    for sub in submissions:
        student = students.get(sub.student_id)
        breakdown.append({
            "name":         student.name          if student else "Unknown",
            "enrollment":   student.enrollment_no if student else "—",
            "status":       sub.status,
            "score":        sub.score,
            "max_score":    sub.max_score,
        })

    total    = len(submissions)
    graded   = sum(1 for s in submissions if s.status == "graded")
    grading  = sum(1 for s in submissions if s.status == "grading")

    return jsonify({
        "assignment_status": assignment.grading_status,
        "total":             total,
        "graded":            graded,
        "grading":           grading,
        "pending":           total - graded - grading,
        "breakdown":         breakdown,
    })


@app.route("/teacher/edit_deadline/<int:assign_id>", methods=["POST"])
@login_required
def edit_deadline(assign_id):
    if current_user.role != "teacher":
        return jsonify({"error": "Unauthorized"}), 403
    assignment = Assignment.query.get_or_404(assign_id)
    if assignment.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    new_date = request.form.get("new_due_date", "").strip()
    if not new_date:
        flash("Please provide a valid date.", "danger")
        return redirect(url_for("teacher_dashboard"))

    from datetime import datetime
    old_date = assignment.due_date
    assignment.due_date = new_date

    try:
        old_dt = datetime.strptime(old_date, "%Y-%m-%d")
        new_dt = datetime.strptime(new_date, "%Y-%m-%d")
        if new_dt > old_dt and assignment.grading_status == "completed":
            assignment.grading_status = "pending"
    except ValueError:
        pass

    db.session.commit()
    flash(f"Deadline updated to {new_date}.", "success")
    return redirect(url_for("teacher_dashboard"))


# ══════════════════════════════════════════════════════════
#  STUDENT
# ══════════════════════════════════════════════════════════

@app.route("/student")
@login_required
def student_dashboard():
    if current_user.role != "student":
        return "Access denied", 403
    assignments = Assignment.query.all()
    submissions = Submission.query.filter_by(student_id=current_user.id).all()
    sub_dict    = {s.assignment_id: s for s in submissions}
    return render_template(
        "student/dashboard.html",
        assignments = assignments,
        sub_dict    = sub_dict,
    )


@app.route("/student/submit/<int:assign_id>", methods=["POST"])
@login_required
def submit_assignment(assign_id):
    if current_user.role != "student":
        return "Access denied", 403

    assignment = Assignment.query.get_or_404(assign_id)

    if assignment.is_past_deadline:
        flash("Deadline has passed. Submissions are closed.", "danger")
        return redirect(url_for("student_dashboard"))

    if "notebook" not in request.files:
        flash("No file uploaded.", "danger")
        return redirect(url_for("student_dashboard"))

    file = request.files["notebook"]
    if not file or not file.filename:
        flash("No file selected.", "danger")
        return redirect(url_for("student_dashboard"))

    if not allowed_notebook(file.filename):
        flash("Only .ipynb files are accepted.", "danger")
        return redirect(url_for("student_dashboard"))

    filename = secure_filename(
        f"user_{current_user.id}_assign_{assign_id}.ipynb"
    )
    filepath = os.path.join(Config.SUBMISSIONS_FOLDER, filename)
    file.save(filepath)

    from core.ipynb_parser import extract_and_validate_ipynb
    parsed = extract_and_validate_ipynb(filepath)
    if parsed["status"] == "error":
        os.remove(filepath)
        flash(f"Submission rejected: {parsed['message']}", "danger")
        return redirect(url_for("student_dashboard"))

    existing = Submission.query.filter_by(
        student_id=current_user.id, assignment_id=assign_id
    ).first()

    if existing:
        existing.file_path   = filepath
        existing.status      = "submitted"
        existing.score       = None
        existing.result_json = None
        existing.released    = False
    else:
        db.session.add(Submission(
            student_id    = current_user.id,
            assignment_id = assign_id,
            file_path     = filepath,
        ))

    db.session.commit()
    flash("Submission uploaded successfully!", "success")
    return redirect(url_for("student_dashboard"))


@app.route("/student/my_grades")
@login_required
def my_grades():
    if current_user.role != "student":
        return "Access denied", 403
    graded_submissions = Submission.query.filter_by(
        student_id = current_user.id,
        status     = "graded",
        released   = True,
    ).all()
    return render_template(
        "student/my_grades.html",
        submissions = graded_submissions,
    )


# ══════════════════════════════════════════════════════════
#  FILE VIEW & DOWNLOAD
# ══════════════════════════════════════════════════════════

@app.route("/view_submission/<int:sub_id>")
@login_required
def view_submission(sub_id):
    sub        = Submission.query.get_or_404(sub_id)
    assignment = Assignment.query.get(sub.assignment_id)
    student    = User.query.get(sub.student_id)

    if current_user.role == "teacher":
        if assignment.teacher_id != current_user.id:
            return "Access denied", 403
    elif current_user.role == "student":
        if sub.student_id != current_user.id:
            return "Access denied", 403
    else:
        return "Access denied", 403

    from core.ipynb_parser import extract_and_validate_ipynb
    parsed = extract_and_validate_ipynb(sub.file_path)
    code   = parsed.get("code") or "Could not extract code from this notebook."

    return render_template(
        "shared/view_submission.html",
        sub        = sub,
        assignment = assignment,
        student    = student,
        code       = code,
    )


@app.route("/download_submission/<int:sub_id>")
@login_required
def download_submission(sub_id):
    from flask import send_file
    sub        = Submission.query.get_or_404(sub_id)
    assignment = Assignment.query.get(sub.assignment_id)
    student    = User.query.get(sub.student_id)

    if current_user.role == "teacher":
        if assignment.teacher_id != current_user.id:
            return "Access denied", 403
    elif current_user.role == "student":
        if sub.student_id != current_user.id:
            return "Access denied", 403
    else:
        return "Access denied", 403

    filename = f"{student.enrollment_no}_{assignment.title.replace(' ', '_')}.ipynb"
    return send_file(
        sub.file_path,
        as_attachment = True,
        download_name = filename,
    )


# ══════════════════════════════════════════════════════════
#  INIT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email=Config.ADMIN_EMAIL).first():
            admin = User(
                name     = Config.ADMIN_NAME,
                email    = Config.ADMIN_EMAIL,
                password = generate_password_hash(
                               Config.ADMIN_PASSWORD,
                               method="pbkdf2:sha256"),
                role     = "admin",
                status   = "approved",
            )
            db.session.add(admin)
            db.session.commit()
            print(f"Admin account created: {Config.ADMIN_EMAIL}")

    print("Starting AutoGrader on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
