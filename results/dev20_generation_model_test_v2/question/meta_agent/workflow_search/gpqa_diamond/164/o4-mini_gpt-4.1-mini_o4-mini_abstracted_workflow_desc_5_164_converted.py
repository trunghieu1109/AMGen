async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1, Subtask 1: Chain-of-Thought to define terms
    cot1_instruction = (
        "Sub-task 1 (Stage 1): Precisely define 'regular branches' in a polyethylene backbone "
        "(spacing, frequency, branch length) and define the 'essential additional reaction step' "
        "(e.g., chain walking vs insertion vs transfer) required for branching with ethylene-only."
    )
    cot1_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask1_desc = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot1_agent([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot1_agent.id}, defining terms, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 (Stage 1) output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask1_desc['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask1_desc)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2, Subtask 1: Debate to verify Statement 1
    debate2_instruction = (
        "Sub-task 1 (Stage 2): Debate and critically verify Statement 1: 'Such combined systems are already implemented on an industrial scale in the US.' "
        "Use patents, reviews, and industrial reports as evidence. Given solutions to the problem from other agents, consider their opinions as additional advice."
    )
    debate2_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                      for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask2_desc = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate2_instruction,
        "context": ["user query", "output of stage1_subtask1"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate2_agents):
            if r == 0:
                inputs = [taskInfo, thinking1, answer1]
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
            thinking2_i, answer2_i = await agent(inputs, debate2_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
            all_thinking2[r].append(thinking2_i)
            all_answer2[r].append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final2_instruction = (
        "Sub-task 1 (Stage 2) Final Decision: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    thinking2, answer2 = await final_decision2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        final2_instruction, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 1 (Stage 2) output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask2_desc['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask2_desc)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3, Subtask 1: Self-Consistency Chain-of-Thought to compile catalyst–activator combos
    sc3_instruction = (
        "Sub-task 1 (Stage 3): Compile and compare catalyst–activator combinations that achieve chain walking "
        "and regular branching using ethylene only (e.g., Ni/Pd α-diimine, phosphine systems), noting activation pathways and branch control."
    )
    N = self.max_sc
    sc3_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask3_desc = {
        "subtask_id": "stage3_subtask1",
        "instruction": sc3_instruction,
        "context": ["user query", "output of stage2_subtask1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3_i, answer3_i = await sc3_agents[i]([taskInfo, thinking2, answer2], sc3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc3_agents[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_sc3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    sc3_final_inst = (
        "Sub-task 1 (Stage 3) Final Decision: Synthesize and choose the most consistent catalyst–activator comparison."
    )
    thinking3, answer3 = await final_sc3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        sc3_final_inst, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_sc3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 1 (Stage 3) output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask3_desc['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask3_desc)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4, Subtask 1: Reflexion to integrate and finalize classification
    reflect4_instruction = (
        "Sub-task 1 (Stage 4): Integrate the fact-check outcomes (Stage 2) and mechanistic data (Stage 3), "
        "resolve any conflicts via structured critique, and form a consistent classification of all four statements. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt."
    )
    cot4_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic4_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask4_desc = {
        "subtask_id": "stage4_subtask1",
        "instruction": reflect4_instruction,
        "context": ["user query", "all previous outputs"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot4_agent(inputs4, reflect4_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot4_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    critic_inst = (
        "Please review the answer above and criticize where it might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    for i in range(self.max_round):
        feedback4, correct4 = await critic4_agent([taskInfo, thinking4, answer4], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic4_agent.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot4_agent(inputs4, reflect4_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4_agent.id}, refined thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 1 (Stage 4) output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask4_desc['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask4_desc)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs