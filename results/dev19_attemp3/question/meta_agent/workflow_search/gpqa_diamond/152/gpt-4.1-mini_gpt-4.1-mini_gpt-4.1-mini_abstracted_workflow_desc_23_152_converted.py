async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Reactant Analysis and Mechanism Mapping

    # Subtask 1: Analyze and classify each reactant and identify compound C using Debate
    debate_instr_1 = "Sub-task 1: Analyze and classify each given reactant and reagent individually, focusing on their chemical nature, role in Michael addition, and possible reactive sites. Explicitly identify and document the structure of compound C without assumption, using all available information. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_1, answer_1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking_1, answer_1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing reactants and compound C, thinking: {thinking_1.content}; answer: {answer_1.content}")
            all_thinking_1[r].append(thinking_1)
            all_answer_1[r].append(answer_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 1: Synthesize and choose the most consistent analysis of reactants and compound C." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Map Michael addition mechanism for each reaction using SC_CoT
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, map the Michael addition mechanism for each reaction (A, B, and C) by determining the nucleophile, electrophile, and site of attack. Explicitly identify the raw carbon skeleton of the Michael adducts before naming, ensuring correct regiochemistry and numbering."
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, mapping Michael addition mechanisms, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_1, answer_1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent Michael addition mechanism mapping.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Tautomeric Form Determination and Nomenclature Assignment

    # Subtask 1: Determine predominant tautomeric form using Reflexion
    reflect_inst_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3 = "Sub-task 3: Determine the predominant tautomeric form (keto vs enol/hydroxy) of each Michael adduct and related intermediates under the given reaction conditions. Tabulate or illustrate both forms and justify the selection of the major tautomer based on chemical stability and reaction context." + reflect_inst_3
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "stage2_subtask1",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining tautomeric forms, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max_3):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback_3.content}; answer: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining tautomeric form determination, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 2: Assign correct IUPAC numbering and nomenclature using Debate
    debate_instr_4 = "Sub-task 4: Assign correct IUPAC numbering and nomenclature to each Michael adduct product based on the verified carbon skeleton and predominant tautomeric form. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage2_subtask2",
        "instruction": debate_instr_4,
        "context": ["user query", thinking_2.content, answer_2.content, thinking_3.content, answer_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_2, answer_2, thinking_3, answer_3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_2, answer_2, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assigning nomenclature, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo, thinking_2, answer_2, thinking_3, answer_3] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Synthesize and choose the most consistent nomenclature assignment.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Final Evaluation and Selection

    # Subtask 1: Integrate all analyses and select correct multiple-choice answer using Debate
    debate_instr_5 = "Sub-task 5: Integrate all structural, mechanistic, and tautomeric analyses to perform a multi-criteria evaluation of the multiple-choice options. Select the correct identities of products A, B, and C by cross-validating the mechanistic plausibility, correct numbering, and tautomeric form. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "stage3_subtask1",
        "instruction": debate_instr_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final evaluation, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo, thinking_4, answer_4] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
