async def forward_181(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Debate
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    debate_instr1 = (
        "Sub-task 1: Extract and summarize the key elements of the Mott–Gurney equation "
        "and the four validity statements. Given solutions to the problem from other agents, "
        "consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents1 = [
        LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents1):
            if r == 0:
                thinking1_r, answer1_r = await agent([taskInfo], debate_instr1, r, is_sub_task=True)
            else:
                input_info1 = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1_r, answer1_r = await agent(input_info1, debate_instr1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1_r.content}; answer: {answer1_r.content}")
            all_thinking1[r].append(thinking1_r)
            all_answer1[r].append(answer1_r)
    final_instr1 = (
        "Sub-task 1: Extract and summarize the key elements of the Mott–Gurney equation "
        "and the four validity statements. Given all the above thinking and answers, reason over them carefully and provide a final answer.")
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + all_thinking1[-1] + all_answer1[-1], final_instr1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: SC_CoT
    cot_sc_instruction = (
        "Sub-task 2: Based on the summary from Sub-task 1, analyze and classify each condition "
        "(trap-free, single vs two-carrier, contact type, diffusion vs drift) against the assumptions of space-charge-limited current.")
    N = self.max_sc
    cot_agents2 = [
        LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N)
    ]
    possible_answers2 = []
    possible_thinkings2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2)
        possible_thinkings2.append(thinking2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent classification.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Reflexion
    reflect_inst3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better.")
    cot_reflect_instruction = (
        "Sub-task 3: Determine which statement fully matches the ideal SCLC regime required for the Mott–Gurney equation." +
        reflect_inst3
    )
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max3 = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": [
            "user query", "thinking of subtask 1", "answer of subtask 1",
            "thinking of subtask 2", "answer of subtask 2"
        ],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max3):
        feedback3, correct3 = await critic_agent3(
            [taskInfo, thinking3, answer3],
            "Please review and provide the limitations of provided solution. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 4: Debate
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    debate_instr4 = (
        "Sub-task 4: Select and prioritize the single correct validity statement based on multi-criteria "
        "conformity to trap-free, single-carrier, Ohmic contact, and negligible diffusion assumptions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer.")
    debate_agents4 = [
        LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_r, answer4_r = await agent([taskInfo, thinking3, answer3], debate_instr4, r, is_sub_task=True)
            else:
                input_info4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_r, answer4_r = await agent(input_info4, debate_instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_r.content}; answer: {answer4_r.content}")
            all_thinking4[r].append(thinking4_r)
            all_answer4[r].append(answer4_r)
    final_instr4 = (
        "Sub-task 4: Select and prioritize the single correct validity statement based on multi-criteria ... "
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.")
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        final_instr4,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs