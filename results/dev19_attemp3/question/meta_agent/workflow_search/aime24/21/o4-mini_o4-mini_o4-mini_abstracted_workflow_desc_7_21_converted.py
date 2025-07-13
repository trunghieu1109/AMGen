async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Compute vertices and chord segments via SC_CoT
    cot_sc_instruction = "Sub-task 1: Compute the 12 vertices of a regular dodecagon on the unit circle and list all side and diagonal segments as pairs of endpoints."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing vertices and chords, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and select the correct list of vertices and chord segments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: Identify perpendicular direction pairs via SC_CoT
    cot_sc_instruction = "Sub-task 5: Identify the three pairs of perpendicular directions among the six direction classes of the dodecagon (0°,30°,60°,90°,120°,150°)."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction, "context": ["Subtask 1 answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, identifying perpendicular pairs, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 5: Synthesize and confirm the three perpendicular direction pairs.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Enumerate and count rectangles via Debate
    debate_instr = "Sub-task 6: For each perpendicular pair, enumerate candidate chords, compute intersections, and count valid rectangles entirely inside the dodecagon. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = []
    all_answer3 = []
    subtask_desc3 = {"subtask_id": "subtask_6", "instruction": debate_instr, "context": ["Subtask 1 answer","Subtask 5 answer"], "agent_collaboration": "Debate"}
    for agent in debate_agents:
        thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instr, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, computing rectangle count, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking3.append(thinking3)
        all_answer3.append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3 + all_answer3, "Sub-task 6: Given all the above thinking and answers, reason over them carefully and provide the rectangle count for each perpendicular direction pair.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 4: Sum the counts via CoT
    cot_instruction = "Sub-task 7: Sum the rectangle counts from Sub-task 6 over all direction pairs to obtain the total number of rectangles."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_7", "instruction": cot_instruction, "context": ["Subtask 6 answer"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, summing counts, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs