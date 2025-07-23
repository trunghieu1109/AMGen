async def forward_164(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant information from the query, including catalyst types, activators, polymerization conditions, "
        "and the four statements to be evaluated. Ensure comprehensive understanding of the problem context to avoid missing critical details."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent extraction and categorization of information.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the question's phrasing and intent to determine whether it expects a single correct statement or multiple correct statements. "
        "Explicitly identify and flag any ambiguity in the question to prevent premature narrowing of answers."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and decide on the question intent regarding number of correct statements and ambiguity."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Perform a rigorous fact-check and validation of each of the four statements individually against up-to-date chemical knowledge, industrial practice, and literature. "
        "Explicitly challenge high-level assumptions such as industrial implementation of ethylene-only dual catalyst branching systems."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide validated truth assessment for each statement individually."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Analyze the relationships and dependencies among catalyst types, activators, industrial feasibility, and economic considerations to understand how these factors collectively influence the formation of branched polyethylene using only ethylene and a dual catalyst system. "
        "Integrate validated facts and contextualize them to assess the correctness and complementarity of the statements."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": "Sub-task 4: Synthesize relationship analysis with fact-checking results.",
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Cross-validate and reconcile conclusions from the fact-checking and relationship analysis subtasks to ensure internal consistency and correctness. "
        "Reassess any conflicting findings and resolve ambiguities."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide reconciled and internally consistent conclusions on the correctness of the four statements."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results2["thinking"], results2["answer"], results4["thinking"], results4["answer"]],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Synthesize all validated information, ambiguity analysis, and cross-checked conclusions to determine which of the four statements are correct regarding the formation of branched polyethylene using only ethylene and a dual catalyst system. "
        "Explicitly consider the possibility of multiple correct statements and clearly justify the final answer(s) based on evidence and reasoning."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Provide final answer with justification on correctness of the four statements."
    )
    debate_desc6 = {
        "instruction": debate_instruction6,
        "final_decision_instruction": final_decision_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "context_desc": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "temperature": 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6["thinking"], results6["answer"])

    return final_answer, logs
