async def forward_164(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and transform the given information from the query into a structured summary highlighting key entities, constraints, and statements for downstream analysis. "
        "Embed feedback to avoid superficial extraction by ensuring clarity on the specific constraints (e.g., 'regular branches' from ethylene only, dual catalyst system) and the exact wording of each statement to be evaluated."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the chemical and mechanistic feasibility of producing regularly branched polyethylene from ethylene alone using a dual catalyst system, "
        "with special focus on the 'essential additional reaction step' and activator compatibility, particularly the role and limitations of aluminum-based activators. "
        "Incorporate organometallic and polymer chemistry knowledge to critically assess statements (B) and (C). Avoid overgeneralization by grounding analysis in mechanistic details."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for mechanistic feasibility analysis."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Investigate the industrial feasibility of dual catalyst systems producing regularly branched polyethylene from ethylene only in the US, "
        "focusing on statement (A). Include a rigorous search for documented commercial processes with named examples and peer-reviewed sources. "
        "Introduce a 'Devil’s Advocate' approach to find counter-evidence or refutations of statement (A) to prevent confirmation bias and overgeneralization."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Evaluate and decide on the industrial feasibility statement with balanced arguments."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Evaluate economic and catalyst selection considerations, focusing on statement (D) about noble metal catalysts' applicability and cost. "
        "Analyze cost implications and practical usage in industrial contexts, ensuring separation from mechanistic and industrial feasibility subtasks to avoid conflation."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent answer for economic and catalyst cost analysis."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Stage 2 Sub-task 1: Integrate insights from mechanistic feasibility (stage_1.subtask_2), industrial feasibility and counter-evidence (stage_1.subtask_3), "
        "and economic/catalyst cost analysis (stage_1.subtask_4) to rigorously evaluate each of the four statements. "
        "Apply a strict evidence-based voting process, weighing mechanistic, industrial, and economic factors to identify the correct statement regarding the formation of regularly branched polyethylene using only ethylene and a dual catalyst system. "
        "Explicitly avoid confirmation bias by requiring justification for acceptance or rejection of each statement."
    )
    final_decision_instruction5 = (
        "Stage 2 Sub-task 1: Provide a final evidence-based evaluation and select the correct statement."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "context": [
            "user query",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2",
            "thinking of stage_1.subtask_3",
            "answer of stage_1.subtask_3",
            "thinking of stage_1.subtask_4",
            "answer of stage_1.subtask_4"
        ],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
