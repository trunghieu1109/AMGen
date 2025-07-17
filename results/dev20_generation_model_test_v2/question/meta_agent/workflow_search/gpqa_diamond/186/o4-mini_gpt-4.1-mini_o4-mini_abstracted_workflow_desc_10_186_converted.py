async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: Compute apparent magnitudes using SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Compute apparent magnitudes for all targets: look up V magnitudes for Canopus and Polaris; "
        "calculate m = M_V + 5 log10(d/10 pc) for the four stars with M_V=15 at distances 5, 10, 50, 200 pc."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent apparent magnitudes for all stars.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: Convert magnitudes to S/N using Reflexion
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 2: Using the apparent magnitudes from Sub-task 1, convert each star's apparent magnitude into "
        "an expected signal-to-noise ratio per binned pixel for a 1 h exposure with ESPRESSO on VLT." + reflect_inst
    )
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N2 = self.max_round
    cot_inputs2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N2):
        feedback2, correct2 = await critic_agent2([taskInfo, thinking2, answer2],
            "Please review and provide the limitations of the provided S/N calculations. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refined thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Select detectable stars using Debate
    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_instruction3 = (
        "Sub-task 3: Select all stars whose computed S/N ≥ 10 per binned pixel in a 1 h exposure and count them." + debate_instr
    )
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N3 = self.max_round
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Select and count detectable stars. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs