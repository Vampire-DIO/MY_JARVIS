from enum import Enum
from typing import Type, ClassVar, Dict
from pydantic import BaseModel, ValidationError

# ---------- 基类定义 (核心控制逻辑) ----------
class BaseCategory:
    """分类基类，强制子类定义关联的Type枚举"""
    types: ClassVar[Type[Enum]]  # 必须指定关联的枚举类型

    @classmethod
    def validate_type(cls, sub_type: Enum) -> None:
        """验证子类型是否属于当前分类"""
        if not isinstance(sub_type, cls.types):
            raise ValueError(
                f"Invalid sub_type '{sub_type}' for category {cls.__name__}. "
                f"Expected type: {cls.types.__name__}"
            )
            
# ---------- 分类枚举定义 (严格绑定子类型) ----------
class MetadataCategory(BaseCategory, Enum):
    """主分类枚举，每个值绑定特定子类型枚举"""
    USER_WORK_HABIT = ("user_work_habit", "用户工作习惯", "WorkHabitType")
    TRAVEL_PLAN = ("travel_plan", "出行计划", "TravelPlanType")
    WORK_PLAN = ("work_plan", "工作计划", "WorkPlanType")
    STUDY_PLAN = ("study_plan", "学习计划", "StudyPlanType")
    DAILY_CHAT = ("daily_chat", "日常闲聊", "DailyChatType")

    def __new__(cls, value: str, display_name: str, type_class: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.display_name = display_name
        obj.type_class_name = type_class  # 存储关联的Type类名
        return obj

    @property
    def types(self) -> Type[Enum]:
        """动态获取关联的子类型枚举类"""
        return globals()[self.type_class_name]

# ---------- 子类型枚举定义 ----------
class WorkHabitType(Enum):
    WORK_SCHEDULE = "work_schedule"
    TOOL_PREFERENCE = "tool_preference"
    COMMUNICATION_STYLE = "communication_style"

class TravelPlanType(Enum):
    TRANSPORTATION = "transportation"
    ACCOMMODATION = "accommodation"
    DIETARY_NEEDS = "dietary_needs"

class WorkPlanType(Enum):
    MEETING_SCHEDULE = "meeting_schedule"
    TASK_PRIORITY = "task_priority"
    DEADLINE_MGMT = "deadline_mgmt"

class StudyPlanType(Enum):
    STUDY_SCHEDULE = "study_schedule"
    RESOURCE_TYPE = "resource_type"
    LEARNING_GOAL = "learning_goal"

class DailyChatType(Enum):
    HOBBY = "hobby"
    FAMILY_TOPIC = "family_topic"
    CURRENT_EVENTS = "current_events"

# ---------- 数据模型 (使用Pydantic进行校验) ----------
class MemoryContent(BaseModel):
    category: MetadataCategory
    sub_type: Enum
    content: str

    @classmethod
    def create(cls, category: MetadataCategory, sub_type: Enum, content: str) -> 'MemoryContent':
        """创建时自动校验类型匹配"""
        category.validate_type(sub_type)
        return cls(category=category, sub_type=sub_type, content=content)