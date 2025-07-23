async def forward_21(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and summarize the properties of the regular dodecagon, including vertices, sides, "
        "and all diagonals relevant to rectangle formation, explicitly clarifying the geometric setup and notation for vertices and chords. "
        "Input content: [taskInfo]"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Clarify and formalize all constraints on rectangles, including: "
        "(a) rectangle sides must lie exactly on polygon sides or diagonals (not subsets), "
        "(b) rectangle vertices can be polygon vertices or intersection points of diagonals, "
        "(c) all diagonals are considered, and "
        "(d) rectangle sides must be parallel and form right angles. "
        "This subtask explicitly addresses the ambiguity and failure in the previous decomposition regarding allowed vertex positions and side alignments. "
        "Input content: [taskInfo, results_0_1['thinking'], results_0_1['answer']]"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate all chords (polygon sides and diagonals) of the dodecagon, grouped by their directions (angles), "
        "and identify all pairs of perpendicular directions (e.g., 0°–90°, 30°–120°, 60°–150°). "
        "This enumeration must include chord lengths and their endpoints (vertices). "
        "Input content: [taskInfo, results_0_2['thinking'], results_0_2['answer']]"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: For each perpendicular direction pair identified in stage_1.subtask_1, enumerate all pairs of parallel chords in each direction that can serve as opposite sides of rectangles. "
        "Compute all intersection points formed by these chords, including vertex-vertex, vertex-intersection, and intersection-intersection points, explicitly considering all possible rectangle corner candidates. "
        "This subtask addresses the previous failure to consider rectangles with vertices at diagonal intersections and mixed vertex types. "
        "Input content: [taskInfo, results_1_1['thinking'], results_1_1['answer']]"
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    review_instruction_1_3 = (
        "Sub-task 3: Validate the enumeration of candidate rectangles for one chosen perpendicular direction pair by explicitly listing all rectangle corners and verifying right angles and side alignments. "
        "This 'checker' subtask ensures no overreliance on unverified formulas and grounds the counting process in concrete examples, directly addressing the previous failure of black-box formula application. "
        "Input content: [taskInfo, results_1_2['thinking'], results_1_2['answer']]"
    )
    review_desc_1_3 = {
        "instruction": review_instruction_1_3,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_1_3, log_1_3 = await self.review(
        subtask_id="stage_1.subtask_3",
        review_desc=review_desc_1_3
    )
    logs.append(log_1_3)

    cot_instruction_2_1 = (
        "Sub-task 1: For each perpendicular direction pair, count all valid rectangles formed by the enumerated chord pairs and intersection points, ensuring that all constraints from stage_0.subtask_2 are satisfied. "
        "This includes counting rectangles with vertices at polygon vertices, diagonal intersections, and mixed vertex types, explicitly avoiding undercounting or overcounting. "
        "Input content: [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_2['thinking'], results_1_2['answer'], results_1_3['thinking'], results_1_3['answer']]"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_2['thinking'], results_1_2['answer'], results_1_3['thinking'], results_1_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_2_2 = (
        "Sub-task 2: Aggregate counts from all perpendicular direction pairs to produce the total number of rectangles inside the dodecagon whose sides lie on polygon sides or diagonals. "
        "Cross-validate the aggregated count against known partial results or examples to ensure consistency and correctness. "
        "Input content: [taskInfo, results_2_1['thinking'], results_2_1['answer']]"
    )
    review_desc_2_2 = {
        "instruction": review_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.review(
        subtask_id="stage_2.subtask_2",
        review_desc=review_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
