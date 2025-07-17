async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: Extract parameters (SC_CoT)
    cot_sc_instruction_1 = "Sub-task 1: Extract and summarize the given lattice parameters (a, α, β, γ) and target plane indices (hkl) from the query."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents1:
        thinking_i, answer_i = await agent([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent summary for the lattice parameters and target plane indices.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 0: Classify crystal system (SC_CoT)
    cot_sc_instruction_2 = "Sub-task 2: Classify the crystal system based on the extracted parameters and identify that a rhombohedral interplanar-spacing formula is required."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking_j, answer_j = await agent([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_j.content}; answer: {answer_j.content}")
        possible_thinkings2.append(thinking_j)
        possible_answers2.append(answer_j)
    final_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent classification and formula requirement.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 1: Derive general formula (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Derive the general expression for d_{hkl} in a rhombohedral lattice with edge length a and angle α, specialized to the (111) plane." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query","thinking1","answer1","thinking2","answer2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 1: Numeric substitution (CoT)
    cot_instruction_4 = "Sub-task 4: Substitute a=10 Å, α=30°, h=k=l=1 into the derived formula and compute the numerical value of the interplanar spacing d_{111}."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction_4,"context":["user query","thinking3","answer3"],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 2: Compare with options (Debate)
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = "Sub-task 5: Compare the computed interplanar spacing with the given choices and select the closest match." + debate_instr
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N5 = self.max_round
    all_thinking5 = [[] for _ in range(N5)]
    all_answer5 = [[] for _ in range(N5)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction_5,"context":["user query","thinking4","answer4"],"agent_collaboration":"Debate"}
    for r in range(N5):
        for agent in debate_agents5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(inputs5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_agent5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Final selection of closest spacing." + final_instr5, is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs