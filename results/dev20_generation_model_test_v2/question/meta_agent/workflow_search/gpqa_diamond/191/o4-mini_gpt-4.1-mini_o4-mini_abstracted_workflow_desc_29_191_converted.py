async def forward_191(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: Extract parameters using SC_CoT
    cot_sc_instruction = "Sub-task 1: Extract and organize all given geometric parameters R, r, s, L, l, theta and charge +q from the query."
    N = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent extraction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])
    # Sub-task 2: Identify principles using SC_CoT
    cot_sc_instruction2 = "Sub-task 2: Identify and classify relevant physical principles and boundary conditions for a conductor with a cavity and point charge."
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["output of subtask_1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent principles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])
    # Sub-task 3: Derive E field expression using SC_CoT
    cot_sc_instruction3 = "Sub-task 3: Derive the general expression for the electric field at an external point by applying Gauss's law enclosing net charge +q."
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["output of subtask_2"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the correct field derivation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])
    # Sub-task 4: Match with choices using Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction4 = "Sub-task 4: Match the derived expression to provided choices." + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinkings4 = [[] for _ in range(N_max)]
    all_answers4 = [[] for _ in range(N_max)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction4, "context": ["output of subtask_3"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking3, answer3] + all_thinkings4[r-1] + all_answers4[r-1], debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinkings4[r].append(thinking)
            all_answers4[r].append(answer)
    final_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_agent4([taskInfo, thinking3, answer3] + all_thinkings4[-1] + all_answers4[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])
    # Sub-task 5: Final justification using Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 5: Select the correct answer and justify." + reflect_inst
    critic_inst = "Please review the answer above and criticize where it might be wrong. If absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking4, answer4]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction, "context": ["output of subtask_4"], "agent_collaboration": "Reflexion"}
    thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refined thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs