from collections import Counter

async def forward_15(self, taskInfo):
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Define variables and record givens (SC_CoT)
    cot_sc_instruction = (
        "Sub-task 1: Define sets D, G, S, H and variables e1, e2, e3, e4 representing the number of residents owning exactly 1,2,3,4 items; "
        "record given cardinalities |D| = 195, |G| = 367, |S| = 562, |H| = 900 and exact-k counts e2 = 437, e3 = 234."
    )
    N_sc = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                  for _ in range(N_sc)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings0.append(thinking)
        possible_answers0.append(answer)
    final_sc_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_sc_agent(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        "Sub-task 1: Synthesize and choose the most consistent definitions and givens.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    # Stage 1: Derive equations (Debate)
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2 = (
        "Sub-task 2: Using outputs from Sub-task 1, write the system of equations: "
        "1) e1 + e2 + e3 + e4 = 900, and 2) |D| + |G| + |S| + |H| = e1 + 2·e2 + 3·e3 + 4·e4."
        + debate_instr
    )
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    N_round = self.max_round
    all_thinking2 = [[] for _ in range(N_round)]
    all_answer2 = [[] for _ in range(N_round)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "Sub-task 1 thinking", "Sub-task 1 answer"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking, answer = await agent(inputs, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking2[r].append(thinking)
            all_answer2[r].append(answer)
    final_decision_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])

    # Stage 2: Solve for e4 (Reflexion)
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 3: Substitute e2 = 437, e3 = 234 and |D|+|G|+|S|+|H| = 195+367+562+900 into the equations and solve for e4. "
        + reflect_inst
    )
    cot_reflect_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "Sub-task 1 thinking", "Sub-task 1 answer", "Sub-task 2 thinking", "Sub-task 2 answer"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, refined thinking: {thinking3.content}; refined answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs