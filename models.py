import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


# ─────────────────────────────────────────────
#  USER
# ─────────────────────────────────────────────
class User(db.Model, UserMixin):
    __tablename__ = "user"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password      = db.Column(db.String(200), nullable=False)

    # role: 'admin' | 'teacher' | 'student'
    role          = db.Column(db.String(20), nullable=False)

    # Teacher-specific
    # status: 'pending' | 'approved' | 'rejected'  (only relevant for teachers)
    status        = db.Column(db.String(20), nullable=True, default="approved")

    # Student-specific
    enrollment_no = db.Column(db.String(50), unique=True, nullable=True)
    branch        = db.Column(db.String(50), nullable=True)
    batch         = db.Column(db.String(50), nullable=True)

    # Relationships
    assignments   = db.relationship("Assignment", backref="creator", lazy=True)
    submissions   = db.relationship("Submission", backref="student",  lazy=True)

    @property
    def is_approved(self):
        """Teachers must be approved. Admins and students are always active."""
        if self.role in ("admin", "student"):
            return True
        return self.status == "approved"

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"


# ─────────────────────────────────────────────
#  ASSIGNMENT
# ─────────────────────────────────────────────
class Assignment(db.Model):
    __tablename__ = "assignment"

    id           = db.Column(db.Integer, primary_key=True)
    title        = db.Column(db.String(200), nullable=False)
    description  = db.Column(db.Text,        nullable=False)
    due_date     = db.Column(db.String(50),  nullable=False)

    # e.g. "Classification" | "Regression" | "Clustering" | "Custom"
    task_type    = db.Column(db.String(50),  nullable=False)

    # JSON string: { "step_name": { "points": N, "depends_on": [...] }, ... }
    rubric_json  = db.Column(db.Text, nullable=False)

    # Path to teacher-uploaded dataset (optional)
    dataset_path = db.Column(db.String(300), nullable=True)

    # Who created this assignment
    teacher_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Whether grading has been triggered
    grading_status = db.Column(db.String(30), default="pending")
    # pending | grading | completed

    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    submissions  = db.relationship("Submission", backref="assignment", lazy=True)

    @property
    def rubric(self):
        """Return rubric as a Python dict."""
        return json.loads(self.rubric_json)

    @property
    def max_score(self):
        """Sum of all rubric step points."""
        return sum(v["points"] for v in self.rubric.values())

    @property
    def is_past_deadline(self):
        try:
            due = datetime.strptime(self.due_date, "%Y-%m-%d")
            return datetime.utcnow() > due
        except ValueError:
            return False

    def __repr__(self):
        return f"<Assignment {self.title}>"


# ─────────────────────────────────────────────
#  SUBMISSION
# ─────────────────────────────────────────────
class Submission(db.Model):
    __tablename__ = "submission"

    id            = db.Column(db.Integer, primary_key=True)
    student_id    = db.Column(db.Integer, db.ForeignKey("user.id"),         nullable=False)
    assignment_id = db.Column(db.Integer, db.ForeignKey("assignment.id"),   nullable=False)

    file_path     = db.Column(db.String(300), nullable=False)

    # status: 'submitted' | 'grading' | 'graded'
    status        = db.Column(db.String(30), default="submitted")

    # Results (filled after grading)
    score         = db.Column(db.Float,  nullable=True)
    max_score     = db.Column(db.Float,  nullable=True)

    # Full grading result stored as JSON string
    result_json   = db.Column(db.Text, nullable=True)

    # Whether the teacher has released results to the student
    released      = db.Column(db.Boolean, default=False)

    submitted_at  = db.Column(db.DateTime, default=datetime.utcnow)
    graded_at     = db.Column(db.DateTime, nullable=True)

    @property
    def result(self):
        """Return full grading result as Python dict."""
        if self.result_json:
            return json.loads(self.result_json)
        return {}

    @property
    def feedback(self):
        """Shortcut to the feedback section of the result."""
        return self.result.get("feedback", {})

    @property
    def timeline(self):
        """Shortcut to detected step timeline."""
        return self.result.get("extracted_timeline", [])

    @property
    def scores_breakdown(self):
        """Shortcut to per-step scores."""
        return self.result.get("final_scores", {})

    @property
    def penalties(self):
        """Shortcut to penalties list."""
        return self.result.get("system_penalties", [])

    def __repr__(self):
        return f"<Submission student={self.student_id} assign={self.assignment_id} status={self.status}>"
