async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1_sub1 = "Sub-task 1: Determine the stereochemical nature of the products formed from each reaction: epoxidation of (E)- and (Z)-oct-4-ene with mCPBA followed by acid-catalyzed ring opening. Explicitly identify whether each product is meso (achiral with internal symmetry) or racemic (chiral), considering the stereospecificity of anti ring opening. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1_sub1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1_sub1 = self.max_round
    all_thinking_stage1_sub1 = [[] for _ in range(N_max_stage1_sub1)]
    all_answer_stage1_sub1 = [[] for _ in range(N_max_stage1_sub1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1_sub1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1_sub1):
        for i, agent in enumerate(debate_agents_stage1_sub1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1_sub1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1_sub1[r-1] + all_answer_stage1_sub1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1_sub1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing stereochemical nature, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1_sub1[r].append(thinking)
            all_answer_stage1_sub1[r].append(answer)
    final_decision_agent_stage1_sub1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1_sub1([taskInfo] + all_thinking_stage1_sub1[-1] + all_answer_stage1_sub1[-1], "Sub-task 1: Determine stereochemical nature final decision. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_stage1_sub2 = "Sub-task 2: Classify the stereochemical relationships among all products from both reactions, identifying which pairs are enantiomers, which are diastereomers, and which are identical. Map the total number of distinct stereoisomers formed and their stereochemical relationships. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1_sub2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1_sub2 = self.max_round
    all_thinking_stage1_sub2 = [[] for _ in range(N_max_stage1_sub2)]
    all_answer_stage1_sub2 = [[] for _ in range(N_max_stage1_sub2)]
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instr_stage1_sub2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1_sub2):
        for i, agent in enumerate(debate_agents_stage1_sub2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], debate_instr_stage1_sub2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking1, answer1] + all_thinking_stage1_sub2[r-1] + all_answer_stage1_sub2[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1_sub2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying stereochemical relationships, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1_sub2[r].append(thinking)
            all_answer_stage1_sub2[r].append(answer)
    final_decision_agent_stage1_sub2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage1_sub2([taskInfo, thinking1, answer1] + all_thinking_stage1_sub2[-1] + all_answer_stage1_sub2[-1], "Sub-task 2: Classify stereochemical relationships final decision. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_stage2_sub1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage2_sub1 = "Sub-task 1: Analyze the chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column. Explicitly verify and challenge assumptions about the achiral HPLC's ability to separate diastereomers, considering differences in physical properties such as polarity and shape." + reflect_inst_stage2_sub1
    cot_agent_stage2_sub1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2_sub1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage2_sub1 = self.max_round
    cot_inputs_stage2_sub1 = [taskInfo, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_stage2_sub1,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_stage2_sub1(cot_inputs_stage2_sub1, cot_reflect_instruction_stage2_sub1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_sub1.id}, analyzing achiral HPLC behavior, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_stage2_sub1):
        feedback, correct = await critic_agent_stage2_sub1([taskInfo, thinking3, answer3], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2_sub1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage2_sub1.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_stage2_sub1(cot_inputs_stage2_sub1, cot_reflect_instruction_stage2_sub1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_sub1.id}, refining achiral HPLC analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    reflect_inst_stage2_sub2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage2_sub2 = "Sub-task 2: Analyze the chromatographic behavior of the combined product mixture on a chiral HPLC column. Confirm the ability of chiral HPLC to separate enantiomers and diastereomers, and predict the number of peaks expected." + reflect_inst_stage2_sub2
    cot_agent_stage2_sub2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2_sub2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage2_sub2 = self.max_round
    cot_inputs_stage2_sub2 = [taskInfo, thinking2, answer2]
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_stage2_sub2,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_stage2_sub2(cot_inputs_stage2_sub2, cot_reflect_instruction_stage2_sub2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_sub2.id}, analyzing chiral HPLC behavior, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_stage2_sub2):
        feedback, correct = await critic_agent_stage2_sub2([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2_sub2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage2_sub2.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_stage2_sub2(cot_inputs_stage2_sub2, cot_reflect_instruction_stage2_sub2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_sub2.id}, refining chiral HPLC analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_stage3_sub1 = "Sub-task 1: Integrate the chromatographic analyses from both achiral and chiral HPLC to select the correct chromatographic outcome from the provided multiple-choice options. Synthesize all prior stereochemical and chromatographic insights. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3_sub1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3_sub1 = self.max_round
    all_thinking_stage3_sub1 = [[] for _ in range(N_max_stage3_sub1)]
    all_answer_stage3_sub1 = [[] for _ in range(N_max_stage3_sub1)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_sub1,
        "context": ["user query", thinking3, answer3, thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3_sub1):
        for i, agent in enumerate(debate_agents_stage3_sub1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instr_stage3_sub1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking_stage3_sub1[r-1] + all_answer_stage3_sub1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage3_sub1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating chromatographic analyses, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage3_sub1[r].append(thinking)
            all_answer_stage3_sub1[r].append(answer)
    final_decision_agent_stage3_sub1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_stage3_sub1([taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking_stage3_sub1[-1] + all_answer_stage3_sub1[-1], "Sub-task 1: Final integration and answer selection. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
