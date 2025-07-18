async def forward_164(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the key elements from the query: catalyst types (group VIa transition metals, noble metals), "
        "activators (aluminum-based and others), polymerization goals (high-density polymer, branched polymer), and industrial context (US implementation). "
        "Provide detailed reasoning and classification with context from the query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze the relationships and constraints between catalysts, activators, and polymer branching, "
        "including chemical compatibility and economic considerations, based on the output from Sub-task 1. "
        "Consider all possible cases and provide consistent reasoning."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent conclusions about catalyst-activator compatibility, "
        "polymer branching feasibility, and economic factors given all the above thinking and answers."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each of the four statements in the query against the analyzed chemical, industrial, and economic information "
        "from Sub-tasks 1 and 2 to determine their correctness regarding the dual catalyst system for branched polyethylene from ethylene. "
        "Provide arguments supporting or refuting each statement."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Decide the correctness of each statement based on the debate and provide a reasoned evaluation."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Synthesize the evaluation results from Sub-task 3 to identify which single statement is correct, "
        "considering possible overlaps or exclusivity among statements. Provide a clear and concise final conclusion."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the final answer indicating the single correct statement regarding the formation of branched polyethylene "
        "using the dual catalyst system and only ethylene as monomer."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
