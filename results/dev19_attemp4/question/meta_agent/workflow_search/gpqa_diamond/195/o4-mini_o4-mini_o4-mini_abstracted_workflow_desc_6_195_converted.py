async def forward_195(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: Extract and summarize given information (SC_CoT)
    cot_sc_instruction = "Sub-task 0_1: Extract and summarize the given information: mass m, amplitude A, Hooke’s law F = –k x, potential energy formula, relativistic energy relation, and the four candidate expressions for v_max."
    N0 = self.max_sc
    cot_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id": "subtask_0_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N0):
        thinking0, answer0 = await cot_agents0[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision0([taskInfo] + possible_thinkings0 + possible_answers0, 
        "Sub-task 0_1: Synthesize and choose the most consistent summary for the given information.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])
    # Stage 1: Derive energy conservation relation (Debate)
    debate_instruction1 = "Sub-task 1_1: Apply energy conservation between the turning point (x = ±A, v = 0) and the center (x = 0, v = v_max) to derive the general equation linking v_max, k, A, m, and c. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N1 = self.max_round
    all_thinking1 = [[] for _ in range(N1)]
    all_answer1 = [[] for _ in range(N1)]
    subtask_desc1 = {"subtask_id": "subtask_1_1", "instruction": debate_instruction1, "context": ["user query", "summary of stage 0"], "agent_collaboration": "Debate"}
    for r in range(N1):
        for i, agent in enumerate(debate_agents1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction1, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking0, answer0] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo, thinking0, answer0] + all_thinking1[-1] + all_answer1[-1], 
        "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2.1: Algebraic solution for v_max (SC_CoT)
    cot_sc_instruction2 = "Sub-task 2_1: Algebraically solve and simplify the derived energy equation to express v_max in closed form under the constraint k A^2/(2*m*c^2) < 1."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2_1", "instruction": cot_sc_instruction2, "context": ["user query", "energy equation from stage 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, 
        "Sub-task 2_1: Synthesize and choose the most consistent algebraic solution for v_max.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2.2: Compare with choices and select correct one (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2_2: Compare the simplified expression for v_max with the four provided choices and select the correct one." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N3 = self.max_round
    cot_inputs3 = [taskInfo, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_2_2", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"], "agent_collaboration": "Reflexion"}
    for i in range(N3):
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs