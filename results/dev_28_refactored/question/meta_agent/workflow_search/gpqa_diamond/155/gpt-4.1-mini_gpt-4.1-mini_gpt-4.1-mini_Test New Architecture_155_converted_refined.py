async def forward_155(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given chemical and experimental information from the query, "
        "including reaction conditions, stereochemical starting materials, and chromatographic methods used. "
        "Ensure clear identification of the reagents, reaction steps, and analytical techniques to set a solid foundation for stereochemical and chromatographic analysis."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the stereochemical outcome of the epoxidation reactions on (E)- and (Z)-oct-4-ene before aqueous acid treatment. "
        "Explicitly determine the stereochemistry of the epoxide products, including identifying which are chiral racemates and which are meso (achiral) species by checking for internal mirror planes. "
        "This step addresses the previous failure to distinguish meso from racemic epoxides and prevents incorrect stereoisomer counts."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent stereochemical outcome for the epoxidation products before acid treatment."
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
        "Sub-task 3: Analyze the stereochemical consequences of the subsequent aqueous acid treatment on the epoxide products. "
        "Consider and explicitly discuss possible reaction pathways, including epoxide ring opening to diols, and the resulting stereoisomeric outcomes. "
        "Evaluate how these transformations affect the total number and types of stereoisomers present. "
        "This subtask addresses the prior oversight of ignoring acid-induced ring opening and its impact on stereochemistry."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent stereochemical outcome after aqueous acid treatment."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Perform a Reflexion-based critical review of the assumptions and conclusions from subtasks 2 and 3. "
        "Re-examine the stereochemical assignments, the meso vs chiral determinations, and the impact of aqueous acid treatment. "
        "Identify any overlooked chemical or stereochemical complexities and revise the stereoisomer count and nature accordingly. "
        "This step ensures that no critical chemical detail is missed before proceeding to chromatographic analysis."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions of subtasks 2 and 3, and suggest any necessary revisions."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Determine the number and types of stereoisomers present in the combined product mixture after aqueous acid treatment, "
        "incorporating the refined stereochemical insights from stage 1. Distinguish between enantiomers, diastereomers, and meso forms, and enumerate the total stereoisomer count accurately. "
        "This subtask consolidates the stereochemical analysis to prepare for chromatographic behavior prediction."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Synthesize and choose the most consistent stereoisomer count and nature after aqueous acid treatment."
    )
    cot_sc_desc5 = {
        "instruction": cot_sc_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Analyze the chromatographic behavior of the identified stereoisomers on both the achiral reverse-phase HPLC and the chiral HPLC columns. "
        "Explicitly state which stereoisomers co-elute or separate under ideal chromatographic resolution, considering that enantiomers co-elute on achiral columns but separate on chiral columns, while diastereomers separate on both. "
        "This step must leverage the accurate stereoisomer count and nature from subtask 5."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Synthesize and choose the most consistent chromatographic peak pattern on achiral and chiral HPLC columns."
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

    cot_instruction7 = (
        "Sub-task 7: Integrate all previous analyses to predict the number of chromatographic peaks observed in each chromatogram (achiral and chiral HPLC). "
        "Select the correct answer choice based on this prediction. Ensure that the reasoning explicitly references the stereochemical and chromatographic principles established earlier to avoid prior mistakes."
    )
    cot_agent_desc7 = {
        "instruction": cot_instruction7,
        "input": [taskInfo, results6["thinking"], results6["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7, log7 = await self.sc_cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc7,
        n_repeat=self.max_sc
    )
    logs.append(log7)

    final_answer = await self.make_final_answer(results7["thinking"], results7["answer"])
    return final_answer, logs
