async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze each individual reaction step in the given sequences to determine its chemical effect on the benzene ring and substituents, "
        "explicitly identifying the functional group transformations and their directing effects. Incorporate a reaction-feasibility check to flag any high-risk or unrealistic transformations (e.g., direct nitration of aniline without protection) and suggest necessary protection or alternative strategies. "
        "This subtask must avoid assumptions and clearly state the expected regiochemical influence of each reagent."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    N_sc = self.max_sc
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    for i in range(N_sc):
        thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_0_1.id}, iteration {i}, analyzing reaction steps, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
        possible_answers_0_1.append(answer_0_1)
        possible_thinkings_0_1.append(thinking_0_1)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct analysis of individual reaction steps.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_0_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_0_2 = (
        "Sub-task 2: After analyzing each reaction step, explicitly assign and verify the positional numbering of substituents on the benzene ring relative to the target molecule 2-(tert-butyl)-1-ethoxy-3-nitrobenzene. "
        "For each intermediate formed after key steps (e.g., Friedel-Crafts alkylation, nitration, reduction, diazotization, hydrolysis, alkylation), produce a clear substitution map (e.g., C1: substituent, C2: substituent, etc.) and confirm consistency with the target's substitution pattern. "
        "This subtask addresses previous failures in regiochemical tracking and numbering consistency by enforcing rigorous positional validation and iterative correction if discrepancies arise. "
        + reflect_inst_0_2
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_0_2 = self.max_round
    cot_inputs_0_2 = [taskInfo, thinking_0_1, answer_0_1]
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_reflect_instruction_0_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2(cot_inputs_0_2, cot_reflect_instruction_0_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_2.id}, initial positional verification, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    for i in range(N_max_0_2):
        feedback_0_2, correct_0_2 = await critic_agent_0_2(
            [taskInfo, thinking_0_2, answer_0_2],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent_0_2.id}, iteration {i}, feedback: {feedback_0_2.content}; correctness: {correct_0_2.content}")
        if correct_0_2.content.strip() == "True":
            break
        cot_inputs_0_2.extend([thinking_0_2, answer_0_2, feedback_0_2])
        thinking_0_2, answer_0_2 = await cot_agent_0_2(cot_inputs_0_2, cot_reflect_instruction_0_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_2.id}, iteration {i+1}, refined positional verification, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_1_1 = (
        "Sub-task 1: Integrate the detailed effects of individual reaction steps and the verified positional maps of intermediates to reconstruct the overall synthetic pathway for each proposed sequence. "
        "Identify the sequence of functional group transformations, their compatibility, and the impact of directing effects on regioselectivity. Explicitly highlight any inconsistencies or conflicts with the target substitution pattern discovered during positional verification. "
        "This subtask must ensure that the synthetic route logically leads to the target molecule without unrealistic assumptions or overlooked protection steps. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": debate_instr_1_1,
        "context": ["user query", thinking_0_1, answer_0_1, thinking_0_2, answer_0_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent(
                    [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2],
                    debate_instr_1_1, r, is_sub_task=True
                )
            else:
                input_infos_1_1 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instr_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Translate the reconstructed synthetic pathways into predicted intermediate structures with explicit substitution patterns after each key step. "
        "Verify the feasibility and regioselectivity of each transformation in the context of the target molecule, ensuring that the final substitution pattern matches 2-(tert-butyl)-1-ethoxy-3-nitrobenzene. "
        "This subtask must cross-check intermediate structures against the positional maps and flag any deviations or impractical steps, incorporating feedback from the reaction feasibility analysis."
    )
    cot_agents_1_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_2, answer_0_2, thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i](
            [taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1],
            cot_sc_instruction_1_2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, iteration {i}, verifying intermediates, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and choose the most consistent verification of intermediate structures.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_2_1 = (
        "Sub-task 1: Critically analyze and classify each proposed synthetic route option based on the verified regioselectivity, functional group compatibility, reaction feasibility, and likelihood of achieving high yield. "
        "Use the validated intermediate structures and positional assignments to determine which sequence correctly and efficiently leads to 2-(tert-butyl)-1-ethoxy-3-nitrobenzene. "
        "This subtask must explicitly justify the selection or rejection of each option, referencing the positional and feasibility checks to avoid previous errors of misassignment and unrealistic assumptions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent(
                    [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
                    debate_instr_2_1, r, is_sub_task=True
                )
            else:
                input_infos_2_1 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
