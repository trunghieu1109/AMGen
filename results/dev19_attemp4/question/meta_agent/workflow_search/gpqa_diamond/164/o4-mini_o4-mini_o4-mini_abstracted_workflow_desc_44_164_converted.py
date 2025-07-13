async def forward_164(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and classify key information using SC-CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract and classify polymerization context, first catalyst, goal of regular branching, "
        "and summarize the four statements."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N1)]
    possible_answers1 = []
    possible_thinkings1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize most consistent summary of key information and statements.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2, Sub-task 2: Mechanistic validation using Debate
    debate_instr2 = ("Sub-task 2: For statement B and C, evaluate mechanistic literature and patents. "
                     "Which are supported?" +
                     "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    N2 = self.max_round
    all_thinking2 = [[] for _ in range(N2)]
    all_answer2 = [[] for _ in range(N2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr2, r, is_sub_task=True)
            else:
                prev_thinks = all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1] + prev_thinks, debate_instr2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Industrial and economic validation using Debate
    debate_instr3 = ("Sub-task 3: For statements A and D, evaluate industrial implementations and cost data. "
                     "Which are supported?" +
                     "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_thinking3 = [[] for _ in range(N2)]
    all_answer3 = [[] for _ in range(N2)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N2):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1], debate_instr3, r, is_sub_task=True)
            else:
                prev = all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1] + prev, debate_instr3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking1, answer1] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2, Sub-task 4: Synthesize evidence using SC-CoT
    cot_sc_instruction4 = (
        "Sub-task 4: Synthesize mechanistic and industrial evidence from Subtasks 2 and 3, "
        "determine validity of each of the four statements."
    )
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction4,
        "context": ["user query", "Subtask1", "Subtask2", "Subtask3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_agents4[i](
            [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3],
            cot_sc_instruction4,
            is_sub_task=True
        )
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thinkings4.append(thinking4)
        possible_answers4.append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize and choose the most consistent evidence-based assessment.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3, Sub-task 5: Final decision using Reflexion
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 5: Determine which single statement is correct regarding dual-catalyst branching. "
        + reflect_inst
    )
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction,
        "context": ["all previous outputs"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    N5 = self.max_round
    for i in range(N5):
        critic_inst = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback, correct = await critic_agent5([taskInfo, thinking5, answer5], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refine thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs