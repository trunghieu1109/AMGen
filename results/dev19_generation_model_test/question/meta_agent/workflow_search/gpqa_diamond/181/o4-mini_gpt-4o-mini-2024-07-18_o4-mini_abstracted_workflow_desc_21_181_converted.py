async def forward_181(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and classify key assumptions (SC_CoT)
    sc_instruction = (
        "Sub-task 0_1: Extract and classify key assumptions and definitions required for the validity of the Mott–Gurney equation: "
        "trap-free material, single-carrier transport, Ohmic injection, negligible diffusion current."
    )
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask0_desc = {
        "subtask_id": "subtask_0_1",
        "instruction": sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking0_i, answer0_i = await sc_agents[i]([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, extracting assumptions, thinking: {thinking0_i.content}; answer: {answer0_i.content}")
        possible_thinkings.append(thinking0_i)
        possible_answers.append(answer0_i)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent0(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 0_1: Synthesize and choose the most consistent assumptions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask0_desc['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask0_desc)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Evaluate candidate statements (Debate)
    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated evaluation of each candidate statement against the assumptions."
    )
    debate_instruction = (
        "Sub-task 1_1: Evaluate each of the four candidate statements against the extracted assumptions." + debate_instr
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    subtask1_desc = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instruction,
        "context": ["user query", "answer of subtask_0_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking1_i, answer1_i = await agent([taskInfo, answer0], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, answer0] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1_i, answer1_i = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating statements, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
            all_thinking1[r].append(thinking1_i)
            all_answer1[r].append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_f, answer1_f = await final_decision_agent1(
        [taskInfo, answer0] + all_thinking1[-1] + all_answer1[-1],
        "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide which statements are consistent or inconsistent.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_f.content}; answer - {answer1_f.content}")
    subtask1_desc['response'] = {"thinking": thinking1_f, "answer": answer1_f}
    logs.append(subtask1_desc)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Select and justify the single valid statement (Reflexion)
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to select the best statement and justify it."
    )
    reflex_instruction = (
        "Sub-task 2_1: Select the single statement that fully satisfies the Mott-Gurney validity conditions and provide a concise justification." + reflect_inst
    )
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs2 = [taskInfo, answer0, answer1_f]
    subtask2_desc = {
        "subtask_id": "subtask_2_1",
        "instruction": reflex_instruction,
        "context": ["user query", "answer of subtask_0_1", "answer of subtask_1_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent2(cot_inputs2, reflex_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, initial selection and justification, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic_agent2(
            [taskInfo, thinking2, answer2],
            "Please review and criticize limitations of the provided selection. If absolutely correct, output exactly 'True' in correct.",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent2.id}, feedback, thinking: {feedback2.content}; answer: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent2(cot_inputs2, reflex_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refinement, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask2_desc['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask2_desc)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs