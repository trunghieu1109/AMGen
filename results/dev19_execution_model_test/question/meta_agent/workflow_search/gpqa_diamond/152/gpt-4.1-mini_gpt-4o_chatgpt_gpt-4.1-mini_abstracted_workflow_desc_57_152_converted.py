async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = "Sub-task 1: Extract and summarize the key chemical entities, reactants, reagents, and reaction types for each Michael addition reaction (A, B, C). Explicitly identify compound C from the problem context to avoid ambiguity. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing reactants and compound C, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 1: Synthesize and finalize extraction and summary of reactants and compound C." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = "Sub-task 2.1: Based on the output from Sub-task 1, enumerate the possible regiochemical outcomes and IUPAC numbering for the Michael addition products of reactions A, B, and C. Address positional assignments carefully with detailed structural analysis and correct numbering according to IUPAC rules."
    N_sc = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc2_1 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2_1, answer2_1 = await cot_agents_2_1[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, enumerating regiochemical outcomes, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        possible_answers_2_1.append(answer2_1)
        possible_thinkings_2_1.append(thinking2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking1, answer1] + possible_thinkings_2_1 + possible_answers_2_1, "Sub-task 2.1: Synthesize and choose the most consistent regiochemical assignments." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = "Sub-task 2.2: Analyze the tautomeric equilibria and possible keto-enol forms of the Michael adducts under the given reaction conditions (NaOEt/EtOH, MeOH/H3O+, KOH/H2O) for each reaction. Integrate mechanistic reasoning to confirm nucleophile identity, regioselectivity, and stereochemical implications consistent with tautomeric forms and structural assignments."
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc2_2 = {
        "subtask_id": "stage_2.subtask_1_and_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking2_1, answer2_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2_2, answer2_2 = await cot_agents_2_2[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, analyzing tautomeric equilibria and mechanistic reasoning, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
        possible_answers_2_2.append(answer2_2)
        possible_thinkings_2_2.append(thinking2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent_2_2([taskInfo, thinking2_1, answer2_1] + possible_thinkings_2_2 + possible_answers_2_2, "Sub-task 2.2: Synthesize and choose the most consistent tautomeric and mechanistic analysis." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])

    reflect_inst_3_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_1 = "Sub-task 3.1: Perform a Reflexion-based critical reassessment of all assumptions, including compound C identity, regioselectivity, and tautomeric forms, to identify and resolve any remaining ambiguities or inconsistencies before final product assignment. This step aims to prevent premature consensus and ensure subtle mechanistic details are fully accounted for." + reflect_inst_3_1
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_1 = self.max_round
    cot_inputs_3_1 = [taskInfo, thinking2_2, answer2_2]
    subtask_desc3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction_3_1,
        "context": ["user query", thinking2_2, answer2_2],
        "agent_collaboration": "Reflexion"
    }
    thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, critical reassessment, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    for i in range(N_max_3_1):
        critic_inst_3_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback3_1, correct3_1 = await critic_agent_3_1([taskInfo, thinking3_1, answer3_1], critic_inst_3_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_1.id}, providing feedback, thinking: {feedback3_1.content}; answer: {correct3_1.content}")
        if correct3_1.content == "True":
            break
        cot_inputs_3_1.extend([thinking3_1, answer3_1, feedback3_1])
        thinking3_1, answer3_1 = await cot_agent_3_1(cot_inputs_3_1, cot_reflect_instruction_3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, refining reassessment, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instr_3_2 = "Sub-task 3.2: Select the correct product assignments (A, B, C) from the multiple-choice options by integrating the detailed structural, tautomeric, and mechanistic analyses. Justify the selection with explicit reference to the refined reasoning and corrected assumptions from previous subtasks. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instr_3_2,
        "context": ["user query", thinking3_1, answer3_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking3_2, answer3_2 = await agent([taskInfo, thinking3_1, answer3_1], debate_instr_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking3_1, answer3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking3_2, answer3_2 = await agent(input_infos_3_2, debate_instr_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct product assignments, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
            all_thinking_3_2[r].append(thinking3_2)
            all_answer_3_2[r].append(answer3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking3_1, answer3_1] + all_thinking_3_2[-1] + all_answer_3_2[-1], "Sub-task 3.2: Final selection of correct product assignments." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs
